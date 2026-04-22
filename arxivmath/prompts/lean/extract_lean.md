# Task Description

You are constructing benchmark items for research-level mathematics formalization in Lean 4 with Mathlib.

You will be given:
- a paper title,
- the abstract,
- theorem-like entries mined from the full paper text, together with proof metadata.

Extract a single, self-contained, precise mathematical statement that is a strong candidate for formalization in Lean. The statement should satisfy the following criteria:
- **Select Lemma**: it is an explicit lemma from the paper. Thus, it needs to be one of the theorem-like entries mined from the full paper text.
- **Self-contained**: the statement should be adjusted so it does not reference the paper in any way or form.
- **Non-Standard Notation**: Make sure to define any non-standard terms and notation used in the statement.
- **Focus on Minor Results**: strongly prefer self-contained lemmas over theorems, propositions, corollaries, or claims. We want to focus on minor results that are more likely to be approachable for formalization.
- **Minimal Hidden Context**: Prefer statements with standard notation and minimal hidden context.
- **No Prior Citations in Proof**: Reject statements that are clearly quoted background facts or explicitly imported from prior work.

Additionally to the extracted statement, you should provide a proof of the statement that is grounded in the original paper proof. This proof should satisfy the following criteria:
- **Grounded in Original Proof**: the proof should be based on the original paper proof, and can even be identical to it. You should not try to be creative.
- **Citations to Prior Work**: Citations to prior work are not acceptable.
- **References to Other Results**: References to other results from the same paper are acceptable, but only under very strict conditions: the referenced results must themselves be strong candidates for formalization, or simple equations, definitions, or limited references. In particular, any reference to a significant result should be avoided. If such a reference appears, then the lemma is not suited for the benchmark. If minor references appear, retain them exactly as they are in the original proof.

If the paper does not contain a clean self-contained lemma candidate with proof that satisfies all the above criteria, set "keep" to false. It is likely that many papers will not contain any suitable candidates, and that's okay. We are looking for a small number of high-quality benchmark items, not a large quantity of mediocre ones.

## Output Format

Respond only with JSON:

```json
{{
    "keep": "boolean indicating whether to keep this candidate for formalization",
    "statement": "self-contained theorem statement",
    "proof": "proof grounded in the original paper proof",
    "rationale": "why this is a strong formalization candidate"
}}
```

If there are no suitable candidates, output:

```json
{{
    "keep": false,
}}
```

# Title
{title}

# Abstract
{abstract}

# Theorem-Like Entries From Full Paper Text
{theorem_entries}
