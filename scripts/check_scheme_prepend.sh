#!/usr/bin/env bash
# Guard against manual scheme prepending in SDK source code.
#
# The only place that should add an https:// scheme to a host is
# normalize_host() in pinecone/_internal/config.py. Writing
# f"https://{...}" anywhere else bypasses that function and can
# introduce double-scheme bugs or inconsistent behaviour.
#
# Usage:
#   ./scripts/check_scheme_prepend.sh
#   Returns exit code 0 if no violations found, 1 otherwise.

set -euo pipefail

VIOLATIONS=$(grep -rn 'f"https://{' pinecone/ --include='*.py' | grep -v 'config\.py' || true)

if [ -n "$VIOLATIONS" ]; then
    echo "ERROR: Manual scheme prepending found in SDK source code."
    echo ""
    echo "The following files contain f\"https://{ patterns outside config.py:"
    echo "$VIOLATIONS"
    echo ""
    echo "Fix: use normalize_host() from pinecone/_internal/config.py instead of"
    echo "manually prepending 'https://' to host strings."
    exit 1
fi

echo "OK: No manual scheme prepending found."
