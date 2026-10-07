#!/usr/bin/env bash
# Upload perp-mm to GitHub.
#
# Usage:
#   ./scripts/push_to_github.sh                      # uses $GITHUB_TOKEN
#   ./scripts/push_to_github.sh ghp_YOURNEWTOKEN     # or pass the token
#
# The token needs the `repo` scope (classic) or read/write contents +
# administration permissions on a fine-grained PAT. The script creates the
# repository if it does not exist, then pushes.
set -euo pipefail

TOKEN="${1:-${GITHUB_TOKEN:-}}"
if [[ -z "$TOKEN" ]]; then
    echo "error: no token. Set GITHUB_TOKEN or pass it as the first argument." >&2
    exit 1
fi
REPO_NAME="${REPO_NAME:-perp-mm}"
REPO_VISIBILITY="${REPO_VISIBILITY:-public}"   # or: private
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 1) identify the account
LOGIN=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.github.com/user \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('login') or '')")
if [[ -z "$LOGIN" ]]; then
    echo "error: GitHub rejected the token (401). Generate a fresh one at" >&2
    echo "  https://github.com/settings/tokens  (classic token with 'repo' scope," >&2
    echo "  or fine-grained with Contents: Read/Write + Administration: Read/Write)" >&2
    exit 1
fi
echo "authenticated as: $LOGIN"

# 2) create the repository if needed
CODE=$(curl -s -o /tmp/gh_repo.json -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    "https://api.github.com/repos/$LOGIN/$REPO_NAME")
if [[ "$CODE" == "404" ]]; then
    echo "creating $LOGIN/$REPO_NAME ($REPO_VISIBILITY)..."
    CODE=$(curl -s -o /tmp/gh_repo.json -w "%{http_code}" \
        -X POST -H "Authorization: Bearer $TOKEN" \
        -H "Accept: application/vnd.github+json" \
        https://api.github.com/user/repos \
        -d "{\"name\":\"$REPO_NAME\",\"description\":\"Advanced market-making engine + terminal frontend for perp-options-clob: SSVI vol surfaces, vega-approximation option MM, multi-level quoting, HJB/GLFT policies, io_uring feed, zero-dependency Rust TUI\",\"private\":$([ "$REPO_VISIBILITY" = private ] && echo true || echo false)}")
    [[ "$CODE" == "201" ]] || { echo "repo creation failed ($CODE)"; cat /tmp/gh_repo.json; exit 1; }
else
    echo "repository $LOGIN/$REPO_NAME already exists"
fi

# 3) push (token is used via a credential helper, never stored on disk)
cd "$ROOT"
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$LOGIN/$REPO_NAME.git"
git -c credential.helper= -c credential.helper='!f() { echo "username=x-access-token"; echo "password='$TOKEN'"; }; f' \
    push -u origin main
echo
echo "done: https://github.com/$LOGIN/$REPO_NAME"
