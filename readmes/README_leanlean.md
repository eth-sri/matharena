# October LeanLean Evaluation

LeanLean measures proof compression across 32 Lean projects from Palomar. A
coding agent edits a repository to reduce its Lean token count while preserving
the protected results and permitted axioms. The [published data](https://github.com/eth-sri/leanlean-october)
contains the scored baselines and submissions. Aggregate scores are on the
[leaderboard](https://matharena.ai/leanlean/).

## Setup and run

On Linux, install uv, Elan, and rootless Podman. From the repository root, prepare
the pinned projects and build the offline evaluation image.

```bash
bash leanlean/scripts/create.sh prepare
bash scripts/leanlean_prepare_eval.sh \
  "${LEANLEAN_WORK_ROOT:-leanlean/work/oct26}/run" data/leanlean/oct26 leanlean/work/oct26/eval
```

The [preprocessing guide](README_leanlean_preprocessing.md) covers custom
selections and storage settings.

Set the API credential for your model. The following example uses
`OPENROUTER_API_KEY` and runs one attempt on problem 17.

```bash
uv run python scripts/run.py --comp leanlean/october-2026 \
  --models openai/gpt-61-sol --n 1 --problems 17 --workers 1 --container-runtime podman
```

The preflight builds and checks the project before the model clock starts. Each
attempt has 8 CPUs, 64 GiB RAM, a 12-hour deadline, and a $100 model budget.
On reaching the cost limit, the agent has up to 15 minutes within the original
deadline to repair its submission. Results are saved under
`outputs/leanlean/october-2026/`. Omit `--problems` to run the whole benchmark. Use
`--run-root /path/to/run` to freeze inputs and write results separately.

## Verify and score submissions

Use the standard MathArena regrade command for saved runs.

```bash
uv run python scripts/regrade.py --comps leanlean/october-2026 \
  --models openai/gpt-61-sol --problems 17
```

The report is `leanlean-grading/results.csv` under the model's output directory.
Omit `--problems` to grade all saved submissions, or use `--resume` to continue
an interrupted grading job. For a frozen run, point `--comp-configs-dir` and
`--output-dir` at its `configs/competitions` and `outputs` directories.

To grade an edited repository directly, pass its root directory and problem ID.

```bash
uv run leanlean-postprocess --submission /path/to/repository \
  --problem 17 --bundle data/leanlean/oct26 --output leanlean/work/oct26/submission-grade
```

The grader snapshots source files, restores the grading inputs, builds the
submission, and checks it with Comparator. The score measures the reduction in
Lean tokens, excluding comments, imports, the challenge, Lake configuration,
and dependencies. For a failed final submission, the October results use the
newest passing source checkpoint from the final three hours, or zero if no
checkpoint passes. To apply this rule, pass the `run-index-*.json` from the
regrade report to `uv run python -m leanlean.limit_fallback --index`. The
command also takes `--bundle`, `--output`, and `--verified` (the regrade directory).
Pass its `combined-results.json` to `leanlean-postprocess --index` to grade the
selected submissions.

## Command monitoring

New images time shell-tool calls on the host, as in the Gemini problem 17
rerun. Most October runs timed supported process invocations through
in-container wrappers. Build a wrapper image to reproduce those timings.

```bash
uv run leanlean-prepare-eval --command-monitoring wrapper \
  --run "${LEANLEAN_WORK_ROOT:-leanlean/work/oct26}/run" \
  --bundle data/leanlean/oct26 --output leanlean/work/oct26/eval-wrapper
```
