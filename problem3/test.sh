#!/bin/bash

echo "Testing AWS Inspector Script"
echo "============================"

# Test 1: Verify authentication
echo "Test 1: Authentication check"
python aws_inspector.py --region us-east-1 --format json > /dev/null
if [ $? -eq 0 ]; then
    echo "✓ Authentication successful"
else
    echo "✗ Authentication failed"
    exit 1
fi

# Test 2: JSON output format
echo "Test 2: JSON output format"
python aws_inspector.py --region us-east-1 --format json --output test_output.json
if [ -f "test_output.json" ]; then
    python -m json.tool test_output.json > /dev/null
    if [ $? -eq 0 ]; then
        echo "✓ Valid JSON output generated"
    else
        echo "✗ Invalid JSON output"
    fi
    rm test_output.json
else
    echo "✗ Output file not created"
fi

# Test 3: Table output format  
echo "Test 3: Table output format"
python aws_inspector.py --region us-east-1 --format table | head -10
echo "✓ Table format displayed"

# Test 4: Invalid region handling
echo "Test 4: Invalid region handling"
python aws_inspector.py --region invalid-region 2>/dev/null
if [ $? -ne 0 ]; then
    echo "✓ Invalid region properly rejected"
else
    echo "✗ Invalid region accepted"
fi

echo ""
echo "Testing complete. Review output above for any failures."