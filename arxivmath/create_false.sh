uv run python arxivmath/create_queries.py --false --model-config gemini/gemini-31-pro-medium
uv run python arxivmath/verify_queries.py --false --model-config gemini/gemini-31-pro-medium
uv run python arxivmath/fulltext_review.py --false --model-config gemini/gemini-31-pro-medium
uv run python arxivmath/fulltext_review.py --false --model-config gemini/gemini-31-pro-medium --key prior_work_filter --prompt arxivmath/prompts/prompt_false_prior_work_filter.md
uv run python arxivmath/fulltext_review.py --false --model-config gemini/gemini-31-pro-medium --key solid_authors --prompt arxivmath/prompts/prompt_solid_authors.md --enable-web-search --skip-ocr
