#!/usr/bin/env bash
set -euo pipefail

if [[ "${SKIP_PUBLIC_PUBLISH:-}" == "1" ]]; then
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if ! command -v bun >/dev/null 2>&1; then
  echo "[public-publish] bun is required but was not found on PATH." >&2
  exit 1
fi

echo "[public-publish] Building Astro site..."
bun run build

if [[ ! -d dist ]]; then
  echo "[public-publish] dist/ not found after build." >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

cp -R dist/. "$tmp_dir"/

pushd "$tmp_dir" >/dev/null
git init -q
git checkout -q -b public
git add -A

source_sha="$(git -C "$repo_root" rev-parse --short HEAD)"
source_branch="$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)"
commit_msg="Publish site from ${source_branch}@${source_sha}"

author_name="$(git -C "$repo_root" config user.name || true)"
author_email="$(git -C "$repo_root" config user.email || true)"

git \
  -c "user.name=${author_name:-public-publisher}" \
  -c "user.email=${author_email:-public-publisher@example.com}" \
  commit -q -m "$commit_msg"

if [[ "${PUBLIC_PUBLISH_DRY_RUN:-}" == "1" ]]; then
  echo "[public-publish] Dry run mode enabled; skipping push."
  popd >/dev/null
  exit 0
fi

git remote add origin "$(git -C "$repo_root" remote get-url origin)"
SKIP_PUBLIC_PUBLISH=1 git push --force origin HEAD:public
popd >/dev/null

echo "[public-publish] Published dist/ to origin/public"
