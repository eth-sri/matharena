## Axle API Key
Make sure to get an API key from [Axle](https://axle.axiommath.ai/) and set it as an environment variable:

```bash
export AXLE_API_KEY=your_api_key_here
```

## Installing instructions for `loogle` tool

### Install `loogle` for Lean `v4.29.0`
Run these commands from the repository root:

```bash
git clone https://github.com/nomeata/loogle external/loogle
cd external/loogle
git checkout 36960b0
```

Then patch `loogle` to use the final Lean `v4.29.0` release instead of the older release candidate:

1. In `external/loogle/lean-toolchain`, set:
```text
leanprover/lean4:v4.29.0
```

2. In `external/loogle/lakefile.lean`, change the mathlib dependency from `master` to:
```lean
require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "v4.29.0"
```

Then build everything:

```bash
lake update
lake build loogle
lake build LoogleMathlibCache
```

## Activate `loogle`
You must set this environment variable before running the competition:

```bash
export MATHARENA_LOOGLE_DIR=/absolute/path/to/matharena/external/loogle
```

## Verify the install
Verify the Python wrapper from the repository root:

```bash
MATHARENA_LOOGLE_DIR=/absolute/path/to/matharena/external/loogle \
uv run python -c "from matharena.tools.lean_execution import loogle; print(loogle('Nat.add_comm', max_results=3))"
```

You should see formatted output like:

```text
Found 15 declarations mentioning Nat.add_comm.

1. Nat.add_comm :  (n m : ℕ) : n + m = m + n
   from Init.Data.Nat.Basic
```

## Installing instructions for `LeanExplore` tool

```bash
git clone https://github.com/justincasher/lean-explore
cd lean-explore
uv sync --extra local
uv run lean-explore data fetch
```

Activate it with:

```bash
export MATHARENA_LEAN_EXPLORE_DIR=absolute/path/lean-explore
```

Verify from the MathArena repo root:

```bash
uv run python -c "from matharena.tools.lean_execution import lean_explore_search; print(lean_explore_search('continuous function', max_results=3))"
```

## Installing instructions for `Comparator`

Use upstream `leanprover/comparator` tag `v4.29.0`.

```bash
bash arxivmath/scripts/lean/setup_comparator.sh
export PATH="/absolute/path/to/matharena/external/landrun/bin:/absolute/path/to/matharena/external/lean4export/.lake/build/bin:$PATH"
```

This installs:
- `external/comparator`
- `external/lean4export`
- `external/landrun/bin/landrun`
- `external/comparator_project`

Quick check:

```bash
cd external/comparator
lake env .lake/build/bin/comparator --help
```

## Installing instructions for `Aristotle`

Install the SDK in the Python environment you use to run MathArena and set your API key:

```bash
python -m pip install aristotlelib
export ARISTOTLE_API_KEY=...
```

Model config:

```text
configs/models/aristotle/aristotle.yaml
```

Notes:
- this integration creates a fresh temporary Lean project for each problem
- Aristotle is configured to use Lean `v4.28.0`
- MathArena also overrides the Lean checker to `lean-4.28.0` for this model
