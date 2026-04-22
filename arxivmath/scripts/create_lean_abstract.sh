uv run python arxivmath/scripts/shared/create_queries.py --lean --model-config gemini/gemini-31-pro-medium
uv run python arxivmath/scripts/shared/verify_queries.py --lean --model-config gemini/gemini-31-pro-medium
uv run python arxivmath/scripts/lean/formalize_statements.py --model-config openai/gpt-54-medium
uv run python arxivmath/scripts/shared/verify_queries.py --semantic-judge --model-config gemini/gemini-31-pro-medium --key semantic_verification_gemini_31_pro_medium
uv run python arxivmath/scripts/shared/verify_queries.py --semantic-judge --model-config openai/gpt-54-high --key semantic_verification_gpt_54_high
uv run python arxivmath/scripts/shared/fulltext_review.py --lean --model-config gemini/gemini-31-pro-medium --key solid_authors --prompt arxivmath/prompts/shared/solid_authors.md --enable-web-search --skip-ocr
uv run python arxivmath/scripts/shared/fulltext_review.py --lean --model-config gemini/gemini-31-pro-medium --key hidden_condition --prompt arxivmath/prompts/lean/hidden_condition.md
uv run python arxivmath/scripts/shared/fulltext_review.py --lean --model-config gemini/gemini-31-pro-medium --key prior_work_filter --prompt arxivmath/prompts/lean/prior_work_filter.md
uv run python arxivmath/scripts/shared/fulltext_review.py --lean --model-config gemini/gemini-31-pro-medium --key ai_generated --prompt arxivmath/prompts/shared/ai_detection.md
