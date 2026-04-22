# Task Description

You are constructing benchmark items for research-level mathematics formalization in Lean 4 with Mathlib.

You will be given only:
- a paper title
- the paper abstract

Your task is to decide whether the abstract contains a clean theorem-like mathematical claim that can plausibly be formalized in Lean 4 with Mathlib, and if so, extract exactly one such statement.

Keep a candidate only if all of the following hold:
- **Abstract-Grounded**: the statement is directly supported by the abstract. Do not invent details that are not clearly present in the abstract.
- **Theorem-Like**: the extracted statement is an actual mathematical claim, not a vague research direction, motivation, or summary sentence.
- **Self-Contained**: the statement can be read on its own without referring to the paper or the abstract.
- **Mathlib-Plausible**: the mathematical objects and assumptions look standard enough that a faithful Lean 4 Mathlib formalization is plausible.
- **Sufficiently Specific**: the abstract gives enough detail to write a precise theorem statement.

Reject the paper if any of the following holds:
- the abstract is too vague to extract a precise theorem-like statement
- the abstract relies on paper-specific notation or definitions that are not sufficiently explained
- the abstract only states empirical, heuristic, or non-theorem-style claims
- the claim would likely require substantial bespoke infrastructure not already standard in Mathlib

Prefer a central claim from the abstract rather than a minor side remark.

## Output Format

Respond only with JSON:

```json
{{
  "keep": boolean,
  "statement": "self-contained mathematical statement",
  "rationale": "short explanation of why this abstract claim is or is not a good Lean formalization candidate"
}}
```

If no suitable candidate exists, output:

```json
{{
  "keep": false,
  "rationale": "short explanation"
}}
```

# Title
{title}

# Abstract
{abstract}
