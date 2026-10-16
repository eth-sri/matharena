# September ArXivLean Evaluation

The September edition contains 53 prove-or-disprove tasks, available as
[MathArena/arxivlean-0926](https://huggingface.co/datasets/MathArena/arxivlean-0926).
The competition config loads the published dataset directly. Question generation
is not needed to evaluate it.

## Submission format

Keep the supplied definitions and submit exactly one theorem:

```lean
theorem arxivlean_proof : arxivleanConjecture := by
  -- Complete proof of the conjecture.
```

or

```lean
theorem arxivlean_disproof : Not arxivleanConjecture := by
  -- Complete proof of the negation of the entire conjecture.
```

Save the complete submission in `Solution.lean` and return it in one Lean code
block. The final grader compares the submitted theorem and its dependencies
against an immutable challenge; it does not replace the theorem header.
`./check.sh` provides preliminary compilation feedback. Additional assumptions,
changed definitions, `sorry`, `admit`, new axioms, and `native_decide` are not
allowed. Standard Lean axioms `propext`, `Quot.sound`, and `Classical.choice` are
permitted. The prompt and initial placeholder do not reveal whether the conjecture
is true or false.

## Pinned runtime (Linux)

Install the repository's Python dependencies, Git, Elan, Go, and `readelf`.
The host must support Landlock for the comparator sandbox. No Axle API key is
needed for this edition. Build Landrun at the same revision as the legacy setup:

```bash
mkdir -p external/landrun/bin
git clone https://github.com/Zouuup/landrun.git external/landrun/src
git -C external/landrun/src checkout --detach 811cfff51ceaf3d9843708aa6d22e9b84ccac8b4
(cd external/landrun/src && go build -o ../bin/landrun ./cmd/landrun)
```

An existing Landrun installation can be reused. Build the new Lean runtime in
its own directory, leaving earlier editions' runtimes available:

```bash
export MATHARENA_LEAN_RUNTIME_ROOT="$HOME/matharena-runtimes/lean-4.34.1"
uv run python arxivmath/scripts/lean/setup_conjecture_runtime.py \
  --runtime-root "$MATHARENA_LEAN_RUNTIME_ROOT" \
  --elan "$HOME/.elan/bin/elan" \
  --landrun-bin "$PWD/external/landrun/bin/landrun"

export ELAN_HOME="$MATHARENA_LEAN_RUNTIME_ROOT/elan"
export PATH="$ELAN_HOME/toolchains/leanprover--lean4---v4.34.1/bin:$PATH"
```

The setup pins Lean **4.34.1**, mathlib
`d13f23b723b8a846827a245b89c10fc7d3f11612`, comparator
`d03acab154d269c06e60e4de7e4cc85deebff94b`, and lean4export
`076e8e57707e813375e8f9da8bf989799ace9680`. It records built assets in
`runtime.json`. The competition also checks the Lake project and manifest hashes.

Build the solver image. The base image build can be skipped if it is already
installed in the selected engine's storage. For Podman:

```bash
podman build -t matharena-harness-arxivlean:lean4.31 docker/harness-arxivlean
podman build -f docker/harness-arxivlean/Conjecture.Dockerfile \
  -t matharena-harness-arxivlean:lean4.34.1 .
```

For Docker, replace `podman` with `docker` in both build commands. The harness
initializes its pinned mathlib cache on first use and mounts it read-only while
solving. Keep the runtime environment variables set for the host-side comparator.

## Run

Set the API credential required by the selected model config. For example,
`openai/gpt-61-sol` uses `OPENROUTER_API_KEY`:

```bash
uv run python scripts/run.py --comp arxivlean/september \
  --models openai/gpt-61-sol --n 1 --container-runtime podman
```

Use `--container-runtime docker` for Docker. Reduce `concurrent_requests` in the
model config for smaller hosts. The September config imposes a $100 limit per
attempt, estimated from token usage and configured prices, with no overall time
limit. Usage from an in-flight request can arrive after the limit is crossed.
Results are saved under `outputs/arxivlean/september/`; subsequent runs resume
missing results unless `--redo-all` is supplied.
