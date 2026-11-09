# LeanLean October 2026

LeanLean evaluates coding agents on proof compression across 32 Lean repositories from Palomar. Agents edit a full project and try to reduce its Lean token count while preserving the protected results and permitted axioms. The published problem IDs are ordered by baseline token count, largest first.

The baseline and scored source trees, diffs, dependency graphs, and results are available in the [LeanLean October data repository](https://github.com/eth-sri/leanlean-october). See the [leaderboard](https://matharena.ai/leanlean/) for aggregate results.

## Prepare and run an evaluation

Use Linux with rootless Podman, Elan, and uv. Run these commands from the MathArena repository root.

```bash
bash leanlean/scripts/create.sh prepare

run=leanlean/work/oct26/run
bundle="$run/bundle"
bash scripts/leanlean_prepare_eval.sh "$run" "$bundle" leanlean/work/oct26/eval

uv run python scripts/run.py \
  --comp leanlean/october-2026 --models openai/gpt-61-sol --n 1 \
  --container-runtime podman --problems 17 --workers 1
```

Preparation fetches the pinned revisions, strips and verifies the projects, and builds an image with their Lean toolchains and dependency caches. The evaluation preflight builds the initial project and runs Comparator before the model clock starts. Remove `--problems 17` to run all 32 projects. Use `--run-root /path/to/run` to freeze inputs and save results separately. Add `--prepare-only` to inspect the prepared run without starting model requests.

Each October attempt has 8 CPUs, 64 GiB of RAM, a 12-hour deadline, and a $100 model budget. Reaching the cost limit allows up to 15 minutes to salvage the saved submission within the original deadline.

For a four-project smoke test, use `bash scripts/leanlean_demo.sh leanlean/work/demo`, prepare its image using the same commands with `leanlean/work/demo` in place of `leanlean/work/oct26`, and select `--comp leanlean/demo`. See [preprocessing](README_leanlean_preprocessing.md) for storage, parallelism, and custom selections.

## Verify and score submissions

The runner saves a source archive for each attempt for separate grading. Final grading restores the registered grading inputs, builds the submitted project, and runs Comparator and the protected-definition checks. The score is the reduction in Lean tokens, excluding comments, imports, the challenge, Lake configuration, and dependencies. For a submission that fails verification, the October results use the newest passing recorded source state from the final three hours of the attempt. An attempt with no passing state in that window earns zero.

Run `leanlean-postprocess` on a checksum-pinned index keyed by problem ID. Each index entry has `path` and `sha256` fields pointing to one result JSON file. The following commands grade the single-model run above.

```bash
uv run python - outputs/leanlean/october-2026/openai/gpt-61-sol > leanlean/work/oct26/results.json <<'PY'
import hashlib
import json
import sys
from pathlib import Path

results = {}
for path in sorted(Path(sys.argv[1]).resolve().glob("*.json*")):
    problem = path.name.split(".", 1)[0]
    if problem.isdecimal():
        results[problem] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
if not results:
    raise SystemExit("No result files found")
print(json.dumps(results))
PY

uv run leanlean-postprocess \
  --index leanlean/work/oct26/results.json --bundle "$bundle" \
  --output leanlean/work/oct26/verification \
  --token-module third_party/lean-lean/src/leanlean/metrics/tokens.py
```

Use `uv run python -m leanlean.limit_fallback --help` to apply the checkpoint rule to the original attempt history. Keep the evaluation image, saved results, source archives, and telemetry through verification.

## Command monitoring

Most published October attempts used **wrapper monitoring**, which recorded supported process invocations inside the container. The Gemini P17 accounting rerun used **host monitoring**. Its timing breakdown was reconstructed from Antigravity's native tool starts and durations. Compound shell commands are grouped as a single tool call in that breakdown.

Fresh images default to host monitoring. It measures whole shell-tool calls, including transport overhead. Wrapper mode can report timings for supported nested invocations, so the two timing granularities should be reported separately. Both modes also support source checkpoints, CPU, and RAM monitoring.

Prepare a wrapper image to collect historical-style process timings.

```bash
uv run leanlean-prepare-eval --run "$run" --bundle "$bundle" \
  --output leanlean/work/oct26/eval-wrapper --command-monitoring wrapper
```

The runner uses the backend recorded in the prepared image's runtime manifest. See [harness monitoring](README_harness.md#monitoring-and-preflight) for the configuration and trace formats.
