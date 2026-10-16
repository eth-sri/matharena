import pytest

from matharena.arxivmath_source import validate_parser_safe_answer as check


@pytest.mark.parametrize("answer", [r"\frac{n(n+1)}{2}", r"\sqrt{2}+1", r"\max\{1,p-1\}", r"\{1,n-1\}", r"(2,2)", r"[0,1]"])
def test_arxiv_answer_parser_safety_accepts_simple_parser_answers(answer):
    assert check(answer)["keep"] is True


@pytest.mark.parametrize("answer", [r"\Sigma^1_2", r"\aleph_0", r"\prod_{j=2}^{\ell}\zeta(j)", r"\{-1\}\cup\{z:|z+1|=1\}", r"(\neg\neg K)^\circ", r"M_3\otimes M_5", r"I\vee J"])
def test_arxiv_answer_parser_safety_rejects_notation_heavy_answers(answer):
    assert check(answer)["keep"] is False
