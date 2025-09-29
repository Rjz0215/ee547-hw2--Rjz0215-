#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import json
import re
import sys
import os
from datetime import datetime
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "sample_data")
DATA_DIR = os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)

PAPERS_PATH = os.path.join(DATA_DIR, "papers.json")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus_analysis.json")  # optional

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def _tokenize(text: str):
    if not text:
        return []
    return [w.lower() for w in _WORD_RE.findall(text)]

class DataStore:
    def __init__(self):
        self.papers = []                  # raw list
        self.by_id = {}                   # id -> paper
        self.category_counts = {}         # cat -> count

        self.global_freq = Counter()
        self.total_words = 0
        self.unique_words = 0

        self.inverted = defaultdict(dict)  # type: dict[str, dict[str, tuple[int,int]]]

    def load(self):
        # --- load papers.json ---
        if not os.path.isfile(PAPERS_PATH):
            msg = f"Missing required data file: {PAPERS_PATH}"
            print(f"[{now_str()}] ERROR {msg}", flush=True)
            raise FileNotFoundError(msg)

        with open(PAPERS_PATH, "r", encoding="utf-8") as f:
            papers = json.load(f)
        if not isinstance(papers, list):
            raise ValueError("papers.json must be a JSON array")

        self.papers = papers
        self.by_id.clear()
        self.category_counts.clear()
        self.global_freq.clear()
        self.inverted.clear()

        for p in self.papers:
            pid = p.get("arxiv_id") or p.get("id")
            if not pid:
                continue
            p["arxiv_id"] = pid
            # 规范化 categories
            cats = p.get("categories") or []
            if isinstance(cats, str):
                cats = [cats]
            p["categories"] = cats
            for c in cats:
                self.category_counts[c] = self.category_counts.get(c, 0) + 1
            title = p.get("title") or ""
            abstract = p.get("abstract") or ""
            title_lc = title.lower()
            abs_lc = abstract.lower()
            p["_title_lc"] = title_lc
            p["_abstract_lc"] = abs_lc

            t_cnt = Counter(_tokenize(title_lc))
            a_cnt = Counter(_tokenize(abs_lc))

            keys = set(t_cnt.keys()) | set(a_cnt.keys())
            for term in keys:
                self.inverted[term][pid] = (t_cnt.get(term, 0), a_cnt.get(term, 0))

            self.global_freq.update(t_cnt)
            self.global_freq.update(a_cnt)

            self.by_id[pid] = p

        if os.path.isfile(CORPUS_PATH):
            try:
                with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                    corpus = json.load(f)
                freqs = corpus.get("word_frequencies")
                if isinstance(freqs, dict):
                    self.global_freq = Counter({str(k): int(v) for k, v in freqs.items()})
            except Exception as e:
                print(f"[{now_str()}] WARN Failed to load corpus_analysis.json: {e}", flush=True)

        self.unique_words = len(self.global_freq)
        self.total_words = int(sum(self.global_freq.values()))

    def abstract_stats(self, p):
        stats = p.get("abstract_stats")
        if isinstance(stats, dict) and all(k in stats for k in ("total_words","unique_words","total_sentences")):
            return stats
        abs_txt = p.get("_abstract_lc", "")
        words = _tokenize(abs_txt)
        sentences = [s for s in re.split(r"[.!?]+", p.get("abstract") or "") if s.strip()]
        return {
            "total_words": len(words),
            "unique_words": len(set(words)),
            "total_sentences": len(sentences),
        }

    def search(self, query: str):
        terms = [t for t in _tokenize(query) if t]
        if not terms:
            return None, "Malformed query: no valid terms"

        doc_sets = []
        for t in terms:
            posting = self.inverted.get(t)
            if not posting:
                return [], None  
            doc_sets.append(set(posting.keys()))


        candidate_ids = set.intersection(*doc_sets) if doc_sets else set()


        results = []
        for pid in candidate_ids:
            score = 0
            title_hits = 0
            abs_hits = 0
            for t in terms:
                tf_title, tf_abs = self.inverted[t][pid]
                title_hits += tf_title
                abs_hits += tf_abs
            score = title_hits + abs_hits
            matches_in = []
            if title_hits > 0:
                matches_in.append("title")
            if abs_hits > 0:
                matches_in.append("abstract")
            paper = self.by_id.get(pid, {})
            results.append({
                "arxiv_id": pid,
                "title": paper.get("title"),
                "match_score": int(score),
                "matches_in": matches_in,
            })

        results.sort(key=lambda r: (-r["match_score"], r["arxiv_id"] or ""))
        return results, None


DATA = DataStore()
try:
    DATA.load()
    print(f"[{now_str()}] INFO Loaded {len(DATA.papers)} papers from {PAPERS_PATH}", flush=True)
except Exception as e:
    print(f"[{now_str()}] ERROR Data load failed: {e}", flush=True)

class Handler(BaseHTTPRequestHandler):
    server_version = "ArxivSimpleHTTP/1.1"

    def _write_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log(self, path, status, extra=""):
        print(f"[{now_str()}] {self.command} {path} - {status}{(' ' + extra) if extra else ''}", flush=True)

    def do_GET(self):
        try:
            if not DATA.papers:
                self._write_json(500, {"error": "Server data not available"})
                self._log(self.path, "500 Internal Server Error")
                return

            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)

            if path == "/papers":
                items = [{
                    "arxiv_id": p.get("arxiv_id"),
                    "title": p.get("title"),
                    "authors": p.get("authors") or [],
                    "categories": p.get("categories") or [],
                } for p in DATA.papers]
                self._write_json(200, items)
                self._log(self.path, "200 OK", f"({len(items)} results)")
                return

            if path.startswith("/papers/"):
                parts = path.split("/", 2)
                arxiv_id = parts[2] if len(parts) == 3 else ""
                paper = DATA.by_id.get(arxiv_id)
                if not paper:
                    self._write_json(404, {"error": "Paper not found", "arxiv_id": arxiv_id})
                    self._log(self.path, "404 Not Found")
                    return
                resp = {
                    "arxiv_id": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "authors": paper.get("authors") or [],
                    "abstract": paper.get("abstract") or "",
                    "categories": paper.get("categories") or [],
                    "published": paper.get("published") or paper.get("published_at") or "",
                    "abstract_stats": DATA.abstract_stats(paper),
                }
                self._write_json(200, resp)
                self._log(self.path, "200 OK")
                return

            if path == "/search":
                q = (qs.get("q", [""])[0] or "").strip()
                if not q:
                    self._write_json(400, {"error": "Missing or empty 'q' parameter"})
                    self._log(self.path, "400 Bad Request")
                    return
                results, err = DATA.search(q)
                if err is not None:
                    self._write_json(400, {"error": err})
                    self._log(self.path, "400 Bad Request")
                    return
                self._write_json(200, {"query": q, "results": results})
                self._log(self.path, "200 OK", f"({len(results)} results)")
                return

            if path == "/stats":
                freq_items = sorted(DATA.global_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
                top_10 = [{"word": w, "frequency": int(c)} for w, c in freq_items]
                resp = {
                    "total_papers": len(DATA.papers),
                    "total_words": int(DATA.total_words),
                    "unique_words": int(DATA.unique_words),
                    "top_10_words": top_10,
                    "category_distribution": {k: int(v) for k, v in sorted(DATA.category_counts.items())},
                }
                self._write_json(200, resp)
                self._log(self.path, "200 OK")
                return

            self._write_json(404, {"error": "Endpoint not found"})
            self._log(self.path, "404 Not Found")

        except Exception as e:
            self._write_json(500, {"error": "Internal server error", "detail": str(e)})
            self._log(self.path, "500 Internal Server Error")

    def log_message(self, format, *args):
        return

def parse_port():
    port = 8080
    if len(sys.argv) >= 2 and sys.argv[1]:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"[{now_str()}] WARN Invalid port '{sys.argv[1]}', falling back to 8080", flush=True)
            port = 8080
    return port

def main():
    port = parse_port()
    host = "0.0.0.0"
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[{now_str()}] INFO Starting ArXiv API server on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"[{now_str()}] INFO Server stopped", flush=True)

if __name__ == "__main__":
    main()
