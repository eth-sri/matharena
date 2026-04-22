#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
COMPARATOR_DIR="${ROOT_DIR}/external/comparator"
EXPORT_DIR="${ROOT_DIR}/external/lean4export"
LANDRUN_DIR="${ROOT_DIR}/external/landrun/bin"
LANDRUN_BIN="${LANDRUN_DIR}/landrun"
PROJECT_DIR="${ROOT_DIR}/external/comparator_project"
LEAN_TOOLCHAIN="leanprover/lean4:v4.29.0"
LANDRUN_URL="https://github.com/Zouuup/landrun/releases/download/v0.1.14/landrun-linux-amd64"

[ -d "${COMPARATOR_DIR}/.git" ] || git clone https://github.com/leanprover/comparator "${COMPARATOR_DIR}"
[ -d "${EXPORT_DIR}/.git" ] || git clone https://github.com/leanprover/lean4export "${EXPORT_DIR}"

git -C "${COMPARATOR_DIR}" fetch --tags
git -C "${COMPARATOR_DIR}" checkout v4.29.0

mkdir -p "${LANDRUN_DIR}"
[ -x "${LANDRUN_BIN}" ] || curl -L --fail -o "${LANDRUN_BIN}" "${LANDRUN_URL}"
chmod +x "${LANDRUN_BIN}"

printf '%s\n' "${LEAN_TOOLCHAIN}" > "${EXPORT_DIR}/lean-toolchain"
(cd "${EXPORT_DIR}" && lake update && lake build)
(cd "${COMPARATOR_DIR}" && lake build)

mkdir -p "${PROJECT_DIR}"
printf '%s\n' "${LEAN_TOOLCHAIN}" > "${PROJECT_DIR}/lean-toolchain"
cat > "${PROJECT_DIR}/lakefile.lean" <<'EOF'
import Lake
open Lake DSL

package comparatorcheck where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.29.0"

lean_lib Challenge where
  roots := #[`Challenge]

lean_lib Solution where
  roots := #[`Solution]
EOF

cat > "${PROJECT_DIR}/Challenge.lean" <<'EOF'
import Mathlib

theorem comparator_template_challenge : True := by
  trivial
EOF

cat > "${PROJECT_DIR}/Solution.lean" <<'EOF'
import Mathlib

theorem comparator_template_challenge : True := by
  trivial
EOF

(cd "${PROJECT_DIR}" && lake update && lake build Challenge Solution)

echo "export PATH=\"${LANDRUN_DIR}:${EXPORT_DIR}/.lake/build/bin:\$PATH\""
