MODELS=(
  "gemini/gemini-31-pro"
  "openai/gpt-54"
  "anthropic/opus_46"
  "moonshot/k25"
)

for model in "${MODELS[@]}"; do
  model_name=${model//\//_}
  python -u scripts/run.py --comp euler/euler --models "$model" --n 2 > "${model_name}.out" 2>&1 &
done

wait