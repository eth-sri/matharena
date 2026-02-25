MODEL=$1
DEFAULT_N=4

COMPS=(
  "aime/aime_2025"
  "hmmt/hmmt_feb_2025"
  "hmmt/hmmt_nov_2025"
  "smt/smt_2025"
  "brumo/brumo_2025"
  "cmimc/cmimc_2025"
  "apex/shortlist_2025"
  "arxiv/all"
)

# Per-comp n overrides
declare -A N_VALUES=(
  ["apex/apex_2025"]=16
  # add more overrides here if needed
  ["arxiv/all"]=8
)

for comp in "${COMPS[@]}"; do
  N=${N_VALUES[$comp]:-$DEFAULT_N}

  echo "Running on $comp with model $MODEL (n=$N)"

  python scripts/run.py \
    --comp "$comp" \
    --models "$MODEL" \
    --n "$N"

  python scripts/check.py \
    --comp "$comp" \
    --model-config gemini/gemini-3-flash-low
done

# Run apex/apex_2025 separately (still supports override)
APEX_COMP="apex/apex_2025"
APEX_N=${N_VALUES[$APEX_COMP]:-$DEFAULT_N}

echo "Running on $APEX_COMP with model $MODEL (n=$APEX_N)"

python scripts/run.py \
  --comp "$APEX_COMP" \
  --models "$MODEL" \
  --n "$APEX_N"

python scripts/check.py \
  --comp "$APEX_COMP" \
  --model-config gemini/gemini-3-flash-low
