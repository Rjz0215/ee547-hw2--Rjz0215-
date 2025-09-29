#!/usr/bin/env python3
"""
AWS Resource Inspector
- Auth via standard AWS mechanisms (CLI config or env vars)
- Verifies with sts:GetCallerIdentity
- Inspects IAM users, EC2 instances, S3 buckets, and Security Groups
- Outputs JSON (default) or a readable table
- Handles common errors and retries gracefully
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError, ReadTimeoutError

# -----------------------------
# Utilities
# -----------------------------

ISO = "%Y-%m-%dT%H:%M:%SZ"

def iso(ts) -> Optional[str]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).strftime(ISO)
    # boto3 sometimes returns tz-aware datetime already
    try:
        return ts.strftime(ISO)
    except Exception:
        return str(ts)

def warn(msg: str) -> None:
    print(f"[WARNING] {msg}", file=sys.stderr)

def err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)

def retryable_call(fn, *args, **kwargs):
    """Run a boto3 call, retrying once on transient network/timeout/throttling."""
    for attempt in range(2):
        try:
            return fn(*args, **kwargs)
        except (EndpointConnectionError, ReadTimeoutError, ConnectionClosedError) as e:
            if attempt == 0:
                warn(f"Transient network error '{e.__class__.__name__}': retrying once...")
                time.sleep(1.0)
                continue
            raise
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"Throttling", "ThrottlingException", "RequestLimitExceeded"} and attempt == 0:
                warn(f"Rate limited ({code}): retrying once...")
                time.sleep(1.0)
                continue
            raise

def validate_region_or_exit(session, region_arg: Optional[str]) -> str:
    """Validate region string; exit non-zero if invalid."""
    # Use EC2 regions catalog as authoritative list
    all_regions = session.get_available_regions("ec2")
    # If user didn't pass region, prefer session region or env/config default
    region = region_arg or session.region_name
    if not region:
        # Fall back to us-east-1 if nothing is configured
        region = "us-east-1"
    if region not in all_regions:
        err(f"Invalid region '{region}'. Valid examples include: {', '.join(sorted(all_regions)[:8])} ...")
        sys.exit(2)
    return region

def make_session(region: Optional[str]) -> Tuple[Any, str]:
    # Beef up retries to be resilient but fast
    cfg = Config(
        region_name=region,
        retries={"max_attempts": 5, "mode": "standard"},
        read_timeout=10,
        connect_timeout=5,
    )
    session = boto3.session.Session(region_name=region)
    # region will be validated later
    return session, cfg

# -----------------------------
# Collectors
# -----------------------------

def get_account_identity(sts_client) -> Dict[str, str]:
    try:
        ident = retryable_call(sts_client.get_caller_identity)
        return {
            "account_id": ident.get("Account"),
            "user_arn": ident.get("Arn"),
        }
    except ClientError as e:
        err("Authentication failed when calling sts:GetCallerIdentity. "
            "Ensure your AWS credentials are configured (aws configure or environment variables).")
        raise

def collect_iam_users(iam_client) -> List[Dict[str, Any]]:
    users: List[Dict[str, Any]] = []
    try:
        paginator = iam_client.get_paginator("list_users")
        for page in retryable_call(paginator.paginate):
            for u in page.get("Users", []):
                username = u.get("UserName")
                user_obj = {
                    "username": username,
                    "user_id": u.get("UserId"),
                    "arn": u.get("Arn"),
                    "create_date": iso(u.get("CreateDate")),
                    "last_activity": None,
                    "attached_policies": [],
                }
                # Try to fetch PasswordLastUsed via GetUser
                try:
                    gu = retryable_call(iam_client.get_user, UserName=username)
                    last_used = gu.get("User", {}).get("PasswordLastUsed")
                    user_obj["last_activity"] = iso(last_used)
                except ClientError as ge:
                    code = ge.response.get("Error", {}).get("Code", "")
                    # AccessDenied -> skip this detail
                    if code in {"AccessDenied", "AccessDeniedException"}:
                        warn(f"Access denied to iam:GetUser for '{username}' - skipping last activity")
                    else:
                        warn(f"Failed iam:GetUser for '{username}': {code}")
                # Attached policies
                try:
                    pols = retryable_call(iam_client.list_attached_user_policies, UserName=username)
                    for p in pols.get("AttachedPolicies", []):
                        user_obj["attached_policies"].append({
                            "policy_name": p.get("PolicyName"),
                            "policy_arn": p.get("PolicyArn"),
                        })
                except ClientError as ge:
                    code = ge.response.get("Error", {}).get("Code", "")
                    if code in {"AccessDenied", "AccessDeniedException"}:
                        warn(f"Access denied to iam:ListAttachedUserPolicies for '{username}' - skipping")
                    else:
                        warn(f"Failed listing attached policies for '{username}': {code}")
                users.append(user_obj)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"AccessDenied", "AccessDeniedException"}:
            warn("Access denied for IAM operations - skipping user enumeration")
            return []
        raise
    return users

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def collect_ec2_instances(ec2_client) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []
    try:
        paginator = ec2_client.get_paginator("describe_instances")
        for page in retryable_call(paginator.paginate):
            for res in page.get("Reservations", []):
                for inst in res.get("Instances", []):
                    inst_obj = {
                        "instance_id": inst.get("InstanceId"),
                        "instance_type": inst.get("InstanceType"),
                        "state": (inst.get("State") or {}).get("Name"),
                        "public_ip": inst.get("PublicIpAddress"),
                        "private_ip": inst.get("PrivateIpAddress"),
                        "availability_zone": (inst.get("Placement") or {}).get("AvailabilityZone"),
                        "launch_time": iso(inst.get("LaunchTime")),
                        "ami_id": inst.get("ImageId"),
                        "ami_name": None,  # fill in later
                        "security_groups": [sg.get("GroupId") for sg in inst.get("SecurityGroups", [])],
                        "tags": {t.get("Key"): t.get("Value") for t in inst.get("Tags", [])} if inst.get("Tags") else {},
                    }
                    instances.append(inst_obj)
        # Resolve AMI names in batches to avoid N calls
        ami_ids = list({i["ami_id"] for i in instances if i.get("ami_id")})
        ami_map = {}
        for chunk in chunked(ami_ids, 100):
            try:
                imgs = retryable_call(ec2_client.describe_images, ImageIds=chunk)
                for img in imgs.get("Images", []):
                    ami_map[img.get("ImageId")] = img.get("Name")
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                # Some accounts may not have permission to describe certain marketplace AMIs
                if code in {"InvalidAMIID.NotFound", "AccessDenied", "UnauthorizedOperation"}:
                    warn(f"Could not describe some AMIs ({code}) - AMI names may be missing")
                else:
                    warn(f"describe_images failed: {code}")
        for inst in instances:
            if inst.get("ami_id"):
                inst["ami_name"] = ami_map.get(inst["ami_id"])
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"AccessDenied", "UnauthorizedOperation"}:
            warn("Access denied for EC2 operations - skipping instances enumeration")
            return []
        raise
    return instances

def _format_port_range(p):
    from_port = p.get("FromPort")
    to_port = p.get("ToPort")
    if from_port is None and to_port is None:
        return "all"
    if from_port == to_port:
        return f"{from_port}-{to_port}"
    return f"{from_port}-{to_port}"

def _ip_ranges(perm) -> List[str]:
    cidrs = [r.get("CidrIp") for r in perm.get("IpRanges", []) if r.get("CidrIp")]
    cidrs6 = [r.get("CidrIpv6") for r in perm.get("Ipv6Ranges", []) if r.get("CidrIpv6")]
    groups = [g.get("GroupId") for g in perm.get("UserIdGroupPairs", []) if g.get("GroupId")]
    res = []
    res.extend(cidrs)
    res.extend(cidrs6)
    res.extend(groups)
    return res or ["-"]

def collect_security_groups(ec2_client) -> List[Dict[str, Any]]:
    sgs: List[Dict[str, Any]] = []
    try:
        paginator = ec2_client.get_paginator("describe_security_groups")
        for page in retryable_call(paginator.paginate):
            for sg in page.get("SecurityGroups", []):
                inbound = []
                for perm in sg.get("IpPermissions", []):
                    proto = perm.get("IpProtocol")
                    proto = "all" if proto in ("-1", None) else proto
                    inbound.append({
                        "protocol": proto,
                        "port_range": _format_port_range(perm),
                        "source": ",".join(_ip_ranges(perm))
                    })
                outbound = []
                for perm in sg.get("IpPermissionsEgress", []):
                    proto = perm.get("IpProtocol")
                    proto = "all" if proto in ("-1", None) else proto
                    outbound.append({
                        "protocol": proto,
                        "port_range": _format_port_range(perm),
                        "destination": ",".join(_ip_ranges(perm))
                    })
                sgs.append({
                    "group_id": sg.get("GroupId"),
                    "group_name": sg.get("GroupName"),
                    "description": sg.get("Description"),
                    "vpc_id": sg.get("VpcId"),
                    "inbound_rules": inbound,
                    "outbound_rules": outbound,
                })
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"AccessDenied", "UnauthorizedOperation"}:
            warn("Access denied for EC2:DescribeSecurityGroups - skipping")
            return []
        raise
    return sgs

def collect_s3_buckets(s3_client, s3_client_regional_factory) -> List[Dict[str, Any]]:
    buckets: List[Dict[str, Any]] = []
    try:
        resp = retryable_call(s3_client.list_buckets)
        for b in resp.get("Buckets", []):
            name = b.get("Name")
            bobj = {
                "bucket_name": name,
                "creation_date": iso(b.get("CreationDate")),
                "region": None,
                "object_count": 0,
                "size_bytes": 0,
            }
            # Region
            try:
                loc = retryable_call(s3_client.get_bucket_location, Bucket=name)
                # us-east-1 returns None or 'us-east-1' depending on API
                loc_constraint = loc.get("LocationConstraint") or "us-east-1"
                bobj["region"] = "us-east-1" if loc_constraint in (None, "", "US") else loc_constraint
            except ClientError as ge:
                code = ge.response.get("Error", {}).get("Code", "")
                warn(f"Failed to get region for bucket '{name}': {code}")
                bobj["region"] = None

            # Count and size (approx) by listing objects in the bucket's region
            try:
                # Use regional client for better latency/consistency
                regional_s3 = s3_client_regional_factory(bobj["region"] or s3_client.meta.region_name)
                paginator = regional_s3.get_paginator("list_objects_v2")
                total = 0
                total_size = 0
                had_contents = False
                for page in retryable_call(paginator.paginate, Bucket=name):
                    contents = page.get("Contents", [])
                    if contents:
                        had_contents = True
                    for obj in contents:
                        total += 1
                        total_size += obj.get("Size", 0)
                if not had_contents:
                    # Empty bucket or no ListBucket permission
                    pass
                bobj["object_count"] = total
                bobj["size_bytes"] = total_size
            except ClientError as ge:
                code = ge.response.get("Error", {}).get("Code", "")
                if code in {"AccessDenied", "AllAccessDisabled"}:
                    err(f"Failed to access S3 bucket '{name}': Access Denied")
                else:
                    warn(f"Failed to list objects in bucket '{name}': {code}")
            buckets.append(bobj)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"AccessDenied", "AccessDeniedException"}:
            warn("Access denied for S3:ListAllMyBuckets - skipping bucket enumeration")
            return []
        raise
    return buckets

# -----------------------------
# Formatting
# -----------------------------

def summarize(iam_users, ec2_instances, s3_buckets, security_groups) -> Dict[str, Any]:
    running_instances = sum(1 for i in ec2_instances if i.get("state") == "running")
    return {
        "total_users": len(iam_users),
        "running_instances": running_instances,
        "total_buckets": len(s3_buckets),
        "security_groups": len(security_groups),
    }

def render_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=False)

def _col(text, width):
    s = "-" if text is None or text == "" else str(text)
    if len(s) > width:
        return s[:width-1] + "…"
    return s.ljust(width)

def render_table(payload: Dict[str, Any]) -> str:
    lines = []
    acct = payload["account_info"]["account_id"]
    region = payload["account_info"]["region"]
    ts = payload["account_info"]["scan_timestamp"]
    lines.append(f"AWS Account: {acct} ({region})")
    lines.append(f"Scan Time: {ts.replace('T', ' ').replace('Z', ' UTC')}")
    lines.append("")

    # IAM USERS
    iam = payload["resources"]["iam_users"]
    lines.append(f"IAM USERS ({len(iam)} total)")
    lines.append(_col("Username", 20) + _col("Create Date", 20) + _col("Last Activity", 20) + _col("Policies", 10))
    for u in iam:
        lines.append(
            _col(u.get("username"), 20) +
            _col((u.get("create_date") or "")[:10], 20) +
            _col((u.get("last_activity") or "")[:10], 20) +
            _col(str(len(u.get("attached_policies", []))), 10)
        )
    lines.append("")

    # EC2
    ec2 = payload["resources"]["ec2_instances"]
    running = sum(1 for i in ec2 if i.get("state") == "running")
    stopped = sum(1 for i in ec2 if i.get("state") == "stopped")
    lines.append(f"EC2 INSTANCES ({running} running, {stopped} stopped)")
    lines.append(
        _col("Instance ID", 20) + _col("Type", 12) + _col("State", 10) +
        _col("Public IP", 16) + _col("Launch Time", 20)
    )
    for i in ec2:
        lines.append(
            _col(i.get("instance_id"), 20) +
            _col(i.get("instance_type"), 12) +
            _col(i.get("state"), 10) +
            _col(i.get("public_ip") or "-", 16) +
            _col((i.get("launch_time") or "")[:16].replace("T", " "), 20)
        )
    lines.append("")

    # S3
    s3s = payload["resources"]["s3_buckets"]
    lines.append(f"S3 BUCKETS ({len(s3s)} total)")
    lines.append(
        _col("Bucket Name", 30) + _col("Region", 12) + _col("Created", 14) +
        _col("Objects", 10) + _col("Size (MB)", 12)
    )
    for b in s3s:
        size_mb = "~{:.1f}".format((b.get("size_bytes", 0) or 0) / (1024 * 1024))
        lines.append(
            _col(b.get("bucket_name"), 30) +
            _col(b.get("region") or "-", 12) +
            _col((b.get("creation_date") or "")[:10], 14) +
            _col(b.get("object_count", 0), 10) +
            _col(size_mb, 12)
        )
    lines.append("")

    # SGs
    sgs = payload["resources"]["security_groups"]
    lines.append(f"SECURITY GROUPS ({len(sgs)} total)")
    lines.append(_col("Group ID", 18) + _col("Name", 18) + _col("VPC ID", 18) + _col("Inbound Rules", 14))
    for sg in sgs:
        lines.append(
            _col(sg.get("group_id"), 18) +
            _col(sg.get("group_name"), 18) +
            _col(sg.get("vpc_id") or "-", 18) +
            _col(len(sg.get("inbound_rules", [])), 14)
        )
    return "\n".join(lines)

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="AWS Resource Inspector")
    parser.add_argument("--region", help="AWS region to inspect (default: from AWS config/env)")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["json", "table"], default="json", help="Output format")
    args = parser.parse_args()

    # Session + region validation
    session, cfg = make_session(args.region)
    # Use STS client without validating region first (global) just to auth;
    # botocore still needs a region, so fall back to session.region_name
    sts_client = session.client("sts", config=cfg)
    try:
        ident = get_account_identity(sts_client)
    except Exception:
        # Error already printed
        sys.exit(1)

    region = validate_region_or_exit(session, args.region)

    # Region-specific clients
    iam_client = session.client("iam", config=cfg)  # global
    ec2_client = session.client("ec2", region_name=region, config=cfg)
    s3_client = session.client("s3", config=cfg)    # list buckets is global

    def s3_regional_factory(rgn: str):
        return session.client("s3", region_name=rgn, config=cfg)

    # Collect
    iam_users = collect_iam_users(iam_client)
    ec2_instances = collect_ec2_instances(ec2_client)
    security_groups = collect_security_groups(ec2_client)
    s3_buckets = collect_s3_buckets(s3_client, s3_regional_factory)

    ts = datetime.now(timezone.utc).strftime(ISO)
    payload = {
        "account_info": {
            "account_id": ident["account_id"],
            "user_arn": ident["user_arn"],
            "region": region,
            "scan_timestamp": ts,
        },
        "resources": {
            "iam_users": iam_users,
            "ec2_instances": ec2_instances,
            "s3_buckets": s3_buckets,
            "security_groups": security_groups,
        },
        "summary": summarize(iam_users, ec2_instances, s3_buckets, security_groups),
    }

    out_text = render_json(payload) if args.format == "json" else render_table(payload)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_text + ("\n" if not out_text.endswith("\n") else ""))
    else:
        print(out_text)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        err("Interrupted by user")
        sys.exit(130)
