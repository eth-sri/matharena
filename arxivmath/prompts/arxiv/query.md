# Task Description

You are constructing evaluation questions for a benchmark on **advanced research-level mathematics**. The benchmark aims to measure whether LLMs are strong enough to rederive **precise mathematical results** from **research papers**, without access to the paper or abstract.

You will be given a **paper title** and **abstract only**. Your task is to determine whether **the central result** of the paper can be converted into a **single, precise, objectively verifiable mathematical question** with a **unique, deterministic answer**.

If such a question can be formed, you must produce it along with its answer. Otherwise, you must reject the paper.
The question must be a difficult research-level mathematics question that requires deep understanding to answer. The question should be interpretable and answerable without access to the original abstract or paper.
Most papers will be rejected, as main research contributions can often not be converted to a question with a single, unambiguous answer.

---

## Criteria for an Acceptable Question–Answer Pair

A paper should be **kept** *only if all* of the following conditions are satisfied:

1. **Direct derivability**  
   The answer must be derivable *directly and unambiguously* from the abstract alone, without requiring access to the full paper or external references.

2. **Main contribution**  
   The question must target a *primary theorem, result, or quantitative claim* of the paper, not background material, motivation, or related work.

3. **Unambiguous and objective**  
   The question must have exactly **one correct answer**, with no dependence on interpretation, conventions, or unstated assumptions.

4. **Non-subjective**  
   The question must not involve opinions, qualitative judgments, or vague descriptors (e.g., "significant," "large," "efficient").

5. **Answer format constraint**  
   The answer must be **either**:
   - a single numerical value, or  
   - a pure LaTeX mathematical expression  

   The answer **must not contain any English words**, including within LaTeX (symbols and variables are allowed).
   Additionally, avoid logical expressions and inequalities. Focus on functions, constants, formulas, or specific mathematical objects.

6. **Question type restriction**  
   The question must **not** be:
   - yes/no  
   - multiple-choice  
   - a request to prove or explain something  

7. **Machine-verifiable**  
   The answer must be suitable for **rule-based verification**, meaning it can be extracted and compared as a string or parsed LaTeX expression.

8. **Self-contained**  
   The question must be understandable *on its own*.  
   - Do **not** reference the paper, authors, or phrases like "in this work."  
   - All notation and quantities used must be explicitly defined in the question.

9. **No paper references in the answer**  
   The answer must be a standalone mathematical object and must not refer to the paper, its results, or its statements.

10. **Claim needs to be proven**
    The authors must say they have actually proven or established the claim in the paper, not just stated it as a conjecture or open problem.

11. **All context provided**
    Ensure the question contains all necessary context from the abstract to be answerable. In particular, all variables, notation, and quantities used in the question must be explicitly defined within the question itself.
    It is okay if questions are long, as long as they remain clear and unambiguous.

12. **Be careful with bounds**
    Some papers prove bounds or inequalities. These are acceptable only if the bound is stated to be tight or exact in the abstract, so that there is a unique correct answer.
    Otherwise, such abstracts should be rejected.
---

## Examples of Unacceptable Questions

- A question that is very easy, and clearly not the main contribution of the paper.
**Example:** In a pilot study of 54 UK high school students taking an assessment of university graduate-level exam questions, the reported pass rate was 82%. What is the pass rate expressed as a decimal?
- A question that contains the answer.
**Example:** Let $c$ be the central charge of a unitary Virasoro CFT$_2$. Define the BTZ threshold dimension by $\Delta_{{\rm BTZ}}:=(c-1)/12$. What is $\Delta_{{\rm BTZ}}$ as a function of $c$? (Answer: \((c-1)/12\))
- A questions whose answer can be easily guessed.
**Example:** A topological space is called \(\kappa\)-resolvable if it contains \(\kappa\) pairwise disjoint dense subsets. Let \(X\) and \(Y\) be regular isodyne topological spaces with \(|X|=|Y|=\omega_1\). In the product space \(X\times Y\), what is the cardinal \(\kappa\) such that \(X\times Y\) is guaranteed to be \(\kappa\)-resolvable?
- A question that is ambiguous. In particular, it refers to "stated" objects in the abstracts which are not available to the reader (who does not have access to the abstract).
**Example:** Consider the exponential Diophantine equation $(2^{{k}}-1)(b^{{k}}-1)=x^{{n}}$ in positive integers $(k,x,n)$ with odd integer parameter $b$. According to the stated result, for which specific odd values of $b$ is it proven that this equation has no positive integer solution $(k,x,n)$? -> the stated result is only given in the abstract, the question itself should be more specific about what "stated result" means.
- A question where the answer contains English words.
**Example:** Let \(\mathcal I\subseteq \mathcal P(\omega)\) be an ideal. Define
\[
 c_{{0,\mathcal I}}:=\bigl\{{x\in \ell_\infty: \forall\varepsilon>0\;\{{n\in\omega: |x_n|\ge \varepsilon\}}\in\mathcal I\bigr\}}.
\]
Let \(K_{{\mathcal I}}:=\operatorname{{Stone}}(\mathcal P(\omega)/\mathcal I)\) be the Stone space of the Boolean algebra \(\mathcal P(\omega)/\mathcal I\). Let \(M(K_{{\mathcal I}})\) be the Banach space of signed Radon measures on \(K_{{\mathcal I}}\), and let \(B_{{M(K_{{\mathcal I}})}}:=\{{\mu\in M(K_{{\mathcal I}}):\|\mu\|\le 1\}}\) be its unit ball, equipped with the weak-* topology (as the dual of \(C(K_{{\mathcal I}})\)).

Write, as a single LaTeX equivalence, the necessary and sufficient condition on \(\mathcal I\) for \(c_{{0,\mathcal I}}\) to be complemented in \(\ell_\infty\). (Answer: \[c_{{0,\mathcal I}}\text{{ is complemented in }}\ell_\infty\ \iff\ B_{{M(K_{{\mathcal I}})}}\text{{ is weak-* separable}}.\])

---

## Output Format

Respond **only** with a JSON object:

```json
{{
  "keep": boolean,
  "question": string,
  "answer": string
}}
```
If no valid question can be formed, output:

```json
{{
  "keep": false
}}
```
If the paper meets all criteria, set "keep": true and include both "question" and "answer".

Do not include any text outside the JSON object.

---

# Title
{title}

# Abstract
{abstract}