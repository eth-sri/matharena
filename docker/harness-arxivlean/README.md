# ArXivLean harness runtime image

This version-neutral image contains:

- Lean 4.31.0
- Node.js 22 and common shell utilities
- SageMath 9.5 and a scientific Python environment
- TeX Live with pdfLaTeX, LuaLaTeX, XeLaTeX, BibTeX and `latexmk`
- Common math packages, recommended fonts, TikZ/TikZ-CD, and extra LaTeX packages
- A separately populated, read-only Mathlib 4.31.0 cache volume

The Codex, Claude Code, Kimi Code, Antigravity CLI, Qwen Code, or OpenCode version
comes from the model config's `harness_version` field. MathArena installs that
exact release into a versioned host cache and mounts it read-only at
`/opt/harness-cli`. If the field is omitted, `latest` is resolved once at run
startup, and the resolved version is recorded in the output history.


Build the image and cache from the repository root:

```bash
scripts/build_harness_arxivlean_image.sh
```

For native Linux Podman, build the same image and cache in Podman's storage:

```bash
scripts/build_harness_arxivlean_image.sh --container-runtime podman
```

Pass `--container-runtime podman` to run/judge commands or shell launchers. Rootless runs
default to `--userns keep-id` to preserve workspace ownership; the build helper
uses the same mapping for its unprivileged smoke test. Docker and Podman image
and cache stores are separate.

The resulting image is `matharena-harness-arxivlean:lean4.31`. All supported
harnesses use the same ephemeral read-only runtime. The current problem
workspace is the only writable bind mount, and Mathlib is mounted read-only.
The image entrypoint sets Linux's `no_new_privs` bit before launching the
selected CLI. This preserves the hardening guarantee on Snap-packaged Docker
versions that reject Docker's equivalent launch-time `--security-opt` flag.

Model traffic does not require general container internet access. MathArena
creates an internal container network for each invocation and exposes only a
host-side proxy. API keys and saved Codex/ChatGPT tokens remain on the host;
the container receives a random invocation-local proxy key.

The per-run `check.sh` gives the coding harness a Lean feedback loop. The
normal MathArena comparator remains responsible for final judging.

The image also includes offline LaTeX tools. Compile a document in `/work` with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -latexoption=-no-shell-escape main.tex
```

Use `-lualatex` or `-xelatex` instead of `-pdf` for the other engines. Packages
are installed at image build time; no downloads are needed during evaluation.
The image build tests all three engines with math packages, TikZ, a local input,
cross-references and a BibTeX bibliography. The build helper repeats those tests
without networking, as an unprivileged user with a read-only root filesystem.

To rebuild only the runtime image, without preparing the Mathlib cache:

```bash
docker build -f docker/harness-arxivlean/Dockerfile \
  -t matharena-harness-arxivlean:lean4.31 docker/harness-arxivlean
```

For September’s Lean 4.34.1 derivative image and pinned comparator, follow the
[September evaluation setup](../../readmes/README_arxivlean_september_evaluation.md).
