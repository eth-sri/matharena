# LeanLean preprocessing

`bash leanlean/scripts/create.sh prepare` fetches the pinned Palomar source
commits, runs lean-strip, and verifies the results with Lean and Comparator.
It links the standard October bundle at `data/leanlean/oct26`. Completed work
is reused on subsequent runs. Use `fetch` to download sources and inspect
`snapshot/index.csv` before stripping.

The setup script uses `uv run --extra leanlean` to install
[lean-strip v1.1.8](https://github.com/eth-sri/lean-strip/tree/ca36ba7aec6cd566bd4e71423a90a615bdd178b0)
at the Git revision pinned in `pyproject.toml`. The strip stage runs that
installation and records its version, commit, and source checksum in
`run/run.json`.

## Custom selections

For a custom selection, set `LEANLEAN_SELECTION` to a selection file and
`LEANLEAN_WORK_ROOT` to a separate workspace. Use `run/bundle/manifest.json`
and `run/bundle/index.csv` to review the verified projects. Point your
competition's `dataset_path` and `harness_config.runtime_manifest` at that
bundle and its generated `eval-runtime.json`.

## Storage and resources

October preprocessing uses `leanlean/work/oct26` by default. Set
`LEANLEAN_WORK_ROOT` to move the workspace and `LEANLEAN_PODMAN_STORAGE_ROOT` to
move Podman's storage separately. Set `LEANLEAN_WORKERS`, `LEANLEAN_CPUS`, and
`LEANLEAN_MEMORY` to control preprocessing parallelism and container limits.
The defaults are six workers, eight CPUs per worker, and 64 GiB RAM per worker.

Image preparation clears its staging directory after a successful build.
LeanLean also compresses the telemetry copy of `model_requests.jsonl` after
each attempt. Keep the workspaces, saved results, and source checkpoints until
grading finishes.

If a rootless Podman container cannot reach the model proxy and the configured
network backend is CNI, use Netavark with a fresh Podman store. Keep the
override set throughout preparation and evaluation.

```bash
mkdir -p leanlean/work
printf '[network]\nnetwork_backend = "netavark"\n' > leanlean/work/containers-netavark.conf
export CONTAINERS_CONF_OVERRIDE="$PWD/leanlean/work/containers-netavark.conf"
```
