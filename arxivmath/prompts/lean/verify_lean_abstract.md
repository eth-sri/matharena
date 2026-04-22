# Verification Task

You are verifying whether an abstract-derived mathematical statement is suitable for a benchmark measuring the ability of LLMs to formalize and prove research-level theorems in Lean 4 with Mathlib.

You are given:
- the paper title
- the abstract
- an extracted theorem-like statement

Keep the statement only if all of the following hold:
- **Faithful to the Abstract**: the statement is genuinely supported by the abstract and does not add important assumptions, definitions, or conclusions that are not clearly present there.
- **Self-Contained**: it can be understood without the rest of the paper.
- **Sufficiently Precise**: it is specific enough that a faithful Lean statement can be written from it.
- **Mathlib-Plausible**: it appears to live in an area and vocabulary that are standard enough for Lean 4 Mathlib.
- **Not Paper-Local**: it does not fundamentally rely on bespoke paper-specific notation or hidden setup.

Reject the statement if the abstract is too underspecified, the extraction overcommits beyond what the abstract says, or the statement likely needs substantial custom infrastructure before it can even be stated faithfully in Lean.

Be strict.

## Output Format

Respond only with JSON:

```json
{{
  "keep": boolean,
  "rationale": "short justification"
}}
```

# Title
{title}

# Abstract
{abstract}

# Extracted Statement
{statement}
