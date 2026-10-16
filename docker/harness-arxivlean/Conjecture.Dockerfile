# Keep the older edition image immutable; reuse its scientific-Python/Sage tools.
ARG BASE_IMAGE=matharena-harness-arxivlean:lean4.31
FROM ${BASE_IMAGE}

ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.34.1
ARG MATHLIB_REV=d13f23b723b8a846827a245b89c10fc7d3f11612

RUN elan toolchain install "${LEAN_TOOLCHAIN}" \
    && elan default "${LEAN_TOOLCHAIN}" \
    && printf '%s\n' "${LEAN_TOOLCHAIN}" > /opt/mathlib-template/lean-toolchain \
    && printf 'import Lake\nopen Lake DSL\n\npackage matharena where\n\nrequire mathlib from git\n  "https://github.com/leanprover-community/mathlib4" @ "%s"\n' "${MATHLIB_REV}" > /opt/mathlib-template/lakefile.lean \
    && lean --version \
    && chmod 0755 /etc/codex \
    && chmod 0644 /etc/codex/requirements.toml
