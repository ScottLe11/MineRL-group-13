#!/bin/bash
# Test runner for ScalarNetwork and network integration tests

echo "========================================="
echo "Running ScalarNetwork Tests"
echo "========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Run tests
python tests/test_scalar_network.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}❌ Some tests failed!${NC}"
    echo -e "${RED}=========================================${NC}"
    exit 1
fi
