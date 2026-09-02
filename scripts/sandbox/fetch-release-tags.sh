#!/usr/bin/env bash
# Populate the current checkout with release tags from the authoritative parent.
# Forks do not inherit tags created after the fork, so fetching `origin` alone
# can leave the install/update E2E matrix empty.

set -euo pipefail

REPO=""
REMOTE="https://github.com/NousResearch/hermes-agent.git"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo)
      [ "$#" -ge 2 ] || { echo 'error: --repo needs a value' >&2; exit 1; }
      REPO="$2"; shift 2 ;;
    --remote)
      [ "$#" -ge 2 ] || { echo 'error: --remote needs a value' >&2; exit 1; }
      REMOTE="$2"; shift 2 ;;
    -h|--help)
      echo 'usage: fetch-release-tags.sh [--repo DIR] [--remote URL]'
      exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$REPO" ]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO="$(git -C "$script_dir" rev-parse --show-toplevel)"
fi

git -C "$REPO" fetch \
  --force \
  --no-tags \
  --no-recurse-submodules \
  "$REMOTE" \
  '+refs/tags/v*:refs/tags/v*'
