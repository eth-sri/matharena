# LeanLean preprocessing

LeanLean follows the same operating pattern as `arxivmath/scripts/create.sh`:
a shell entry point with explicit stages, environment settings for a persistent
workspace, resumable Python stages, and a verified bundle for evaluation.
Palomar provides the selected formalizations, so no model screening or question
generation is needed. Stripping and dependency analysis use lean-strip v1.1.8.

## Prepare a selection

With `uv`, rootless Podman and Elan installed, run from the repository root:

```bash
export LEANLEAN_WORK_ROOT=/absolute/path/to/leanlean-oct26
export LEANLEAN_SELECTION=configs/leanlean/october-2026.json
bash leanlean/scripts/create.sh fetch

# Inspect snapshot/index.csv, then run stripping and verification.
bash leanlean/scripts/create.sh prepare
```

`fetch` is the default mode and only freezes registry metadata and downloads
pinned repositories. `prepare` reruns fetching incrementally, strips source
copies, verifies them and writes `run/bundle/` for review. Matching completed
work is reused. Neither command adds files to `data/` or starts model evaluation.

The default selection contains the 32 October projects; the default workspace
is `leanlean/work/oct26`, which is ignored by Git. Use a separate workspace for
each selection or tool version. Paths are resolved from the repository root.
Settings can be changed without editing the script:

```bash
export LEANLEAN_WORKERS=6
export LEANLEAN_CPUS=8             # Per container.
export LEANLEAN_MEMORY=64g         # Per container.
export LEANLEAN_TIMEOUT=21600      # Seconds per strip/build/check command.
export LEANLEAN_PODMAN_STORAGE_ROOT=/path/to/podman-store
```

Without a storage override, Podman uses `<workspace>/podman`. The script uses
`uv run --extra leanlean` to install the pinned stripping dependency.

## Stripping and verification

The per-project operation is equivalent to:

```bash
lean-strip /path/to/project \
  --comparator /path/to/project/comparator.json \
  --container podman --cpus 8 --memory 64g --jobs 8 \
  --retain-lean-only --clean-certify
```

The wrapper resolves each project's actual Comparator configuration, shares
the upstream container image across workers, and verifies the applied output
with a build and Comparator check. Only accepted projects enter `run/bundle/`.
A failed project leaves the bundle marked incomplete and the command exits
unsuccessfully. Run settings, source hashes and tool fingerprints are recorded
in `run/run.json`.

The implementation lives in `src/leanlean/palomar_source.py` and
`src/leanlean/strip.py`. The Python stages can also be run directly:

```bash
uv run --extra leanlean leanlean-fetch \
  --selection configs/leanlean/october-2026.json --output /path/to/snapshot
uv run --extra leanlean leanlean-strip \
  --snapshot /path/to/snapshot --output /path/to/run --workers 6
```

## Review the bundle

Review `run/bundle/index.csv`, `manifest.json` and the per-project reports before
preparing the evaluation image. The bundle includes stripped sources, reports,
logs, and declaration dependency graphs. The generated
`dependency-graphs.json` indexes the graphs and their checksums. Downloaded package
trees, Lean build outputs, and toolchain caches stay in the workspace for image preparation.

The October selection pins the Palomar entry versions, source commits, and
published problem order. The exact scored baselines are in the
[published data repository](https://github.com/eth-sri/leanlean-october).
See the [evaluation guide](README_leanlean.md) for running and scoring a prepared bundle.
