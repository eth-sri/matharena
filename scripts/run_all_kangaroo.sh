MODEL=$1
COMPS=(
  "kangaroo/kangaroo_2025_1-2"
  "kangaroo/kangaroo_2025_3-4"
  "kangaroo/kangaroo_2025_5-6"
  "kangaroo/kangaroo_2025_7-8"
  "kangaroo/kangaroo_2025_9-10"
  "kangaroo/kangaroo_2025_11-12"
)

for comp in "${COMPS[@]}"; do
  echo "Running on $comp with model $MODEL"
  uv run python3 scripts/run.py \
    --comp "$comp" \
    --n 4 \
    --models "$MODEL"
done
