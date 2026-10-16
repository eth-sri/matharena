#!/usr/bin/env bash
set -euo pipefail

IMAGE="matharena-harness-arxivlean:lean4.31"
CACHE_VOLUME="matharena-lean431-cache"
CONTAINER_RUNTIME="${MATHARENA_CONTAINER_RUNTIME:-docker}"
source "$(dirname "${BASH_SOURCE[0]}")/lib/container_runtime.sh"
parse_container_runtime "$@"
if (( ${#CONTAINER_ARGS[@]} > 0 )); then
  echo "Usage: $0 [--container-runtime docker|podman]" >&2
  exit 2
fi
RUNTIME="${CONTAINER_RUNTIME}"

USERNS_ARGS=()
if [[ -n "${MATHARENA_CONTAINER_USERNS:-}" ]]; then
  USERNS_ARGS=(--userns "${MATHARENA_CONTAINER_USERNS}")
elif [[ "${RUNTIME}" == podman && "$(id -u)" != 0 ]]; then
  USERNS_ARGS=(--userns keep-id)
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

"${RUNTIME}" build \
  --file docker/harness-arxivlean/Dockerfile \
  --tag "${IMAGE}" \
  docker/harness-arxivlean

"${RUNTIME}" volume inspect "${CACHE_VOLUME}" >/dev/null 2>&1 \
  || "${RUNTIME}" volume create "${CACHE_VOLUME}" >/dev/null

LOCAL_LEAN_PROJECT="${REPO_ROOT}/external/comparator_project"
if "${RUNTIME}" run --rm --network none \
  --mount "type=volume,src=${CACHE_VOLUME},dst=/mathlib-cache,readonly" \
  "${IMAGE}" \
  test -f /mathlib-cache/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean
then
  echo "Reusing populated ${CACHE_VOLUME}."
elif [[ -f "${LOCAL_LEAN_PROJECT}/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean" ]]
then
  "${RUNTIME}" run --rm \
    --network none \
    --mount "type=bind,src=${LOCAL_LEAN_PROJECT},dst=/host-cache,readonly" \
    --mount "type=volume,src=${CACHE_VOLUME},dst=/mathlib-cache" \
    "${IMAGE}" \
    sh -c 'cp -a /host-cache/. /mathlib-cache/ && chmod -R a+rX /mathlib-cache'
else
  "${RUNTIME}" run --rm \
    --network bridge \
    --mount "type=volume,src=${CACHE_VOLUME},dst=/mathlib-cache" \
    "${IMAGE}" \
    sh -c '
      set -eu
      cd /mathlib-cache
      if [ ! -f .lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean ]; then
        cp /opt/mathlib-template/lean-toolchain /opt/mathlib-template/lakefile.lean .
        lake update
        lake exe cache get
      fi
      chmod -R a+rX /mathlib-cache
    '
fi

"${RUNTIME}" run --rm --network none "${IMAGE}" lean --version
"${RUNTIME}" run --rm --network none \
  --mount "type=volume,src=${CACHE_VOLUME},dst=/mathlib-cache,readonly" \
  "${IMAGE}" \
  sh -c '
    printf "%s\n" "import Mathlib" "" "example : True := by trivial" > /tmp/Smoke.lean
    export LEAN_PATH="/mathlib-cache/.lake/packages/Cli/.lake/build/lib/lean:/mathlib-cache/.lake/packages/batteries/.lake/build/lib/lean:/mathlib-cache/.lake/packages/Qq/.lake/build/lib/lean:/mathlib-cache/.lake/packages/aesop/.lake/build/lib/lean:/mathlib-cache/.lake/packages/proofwidgets/.lake/build/lib/lean:/mathlib-cache/.lake/packages/importGraph/.lake/build/lib/lean:/mathlib-cache/.lake/packages/LeanSearchClient/.lake/build/lib/lean:/mathlib-cache/.lake/packages/plausible/.lake/build/lib/lean:/mathlib-cache/.lake/packages/mathlib/.lake/build/lib/lean:/mathlib-cache/.lake/build/lib/lean:/elan/toolchains/leanprover--lean4---v4.31.0/lib/lean"
    lean /tmp/Smoke.lean
  '

# Match evaluation restrictions: no network, unprivileged user, read-only root.
"${RUNTIME}" run --rm --network none --read-only --cap-drop ALL \
  "${USERNS_ARGS[@]}" \
  --user "$(id -u):$(id -g)" \
  --tmpfs /work:rw,nosuid,nodev,size=256m,mode=1777 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m,mode=1777 \
  --env HOME=/work/.harness-home \
  "${IMAGE}" sh /opt/harness-tex-smoke/check.sh
