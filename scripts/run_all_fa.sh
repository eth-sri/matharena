MODEL=$1
DEFAULT_N=4

COMPS=(
  "aime/aime_2025"
  "aime/aime_2026"
  "hmmt/hmmt_feb_2026"
  "hmmt/hmmt_feb_2025"
  "hmmt/hmmt_nov_2025"
  "smt/smt_2025"
  "brumo/brumo_2025"
  "cmimc/cmimc_2025"
  "apex/shortlist_2025"
  "arxiv/december"
  "arxiv/january"
  "arxiv/february"
  "apex/apex_2025"
)

# Per-comp n overrides
declare -A N_VALUES=(
  ["apex/apex_2025"]=16
  # add more overrides here if needed
  ["arxiv/december"]=4
  ["arxiv/january"]=4
)

COMP_N_OVERRIDES=()
for comp in "${COMPS[@]}"; do
  if [[ -n "${N_VALUES[$comp]}" ]]; then
    COMP_N_OVERRIDES+=("${comp}=${N_VALUES[$comp]}")
  fi
done

echo "Running on comps: ${COMPS[*]} with model $MODEL (default n=$DEFAULT_N, overrides: ${COMP_N_OVERRIDES[*]})"
python scripts/run.py \
  --comp "${COMPS[@]}" \
  --models "$MODEL" \
  --n "$DEFAULT_N" \
  --comp-n "${COMP_N_OVERRIDES[@]}"

for comp in "${COMPS[@]}"; do
  python scripts/curation/check.py  --comp "$comp" --model-config gemini/gemini-3-flash-low
done
