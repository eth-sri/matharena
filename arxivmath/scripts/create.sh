uv run python arxivmath/scripts/shared/create_queries.py --model-config gemini/gemini-31-pro-medium
uv run python arxivmath/scripts/shared/verify_queries.py --model-config gemini/gemini-31-pro-medium
uv run python arxivmath/scripts/shared/fulltext_review.py --model-config gemini/gemini-31-pro-medium
uv run python arxivmath/scripts/shared/fulltext_review.py --model-config gemini/gemini-31-pro-medium --key solid_authors --prompt arxivmath/prompts/shared/solid_authors.md --enable-web-search --skip-ocr
uv run python arxivmath/scripts/shared/fulltext_review.py --model-config gemini/gemini-31-pro-medium --key prior_work_filter --prompt arxivmath/prompts/arxiv/prior_work_filter.md
uv run python arxivmath/scripts/shared/fulltext_review.py --model-config gemini/gemini-31-pro-medium --key ai_generated --prompt arxivmath/prompts/shared/ai_detection.md
