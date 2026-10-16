"""Offline tests of BrokenArXiv through the shared source stages."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import arxivmath.app as review_app
from arxivmath.scripts.broken import export_false_proofs as exporter
from arxivmath.scripts.source import (
    audit_pipeline,
    investigate_sources,
    screen_abstracts,
    verify_questions,
)
from matharena.arxivbench_utils import extract_json
from matharena.arxivmath_audit import audit_paper
from matharena.arxivmath_source import (
    FALSE_CHECKS,
    FALSE_SCREEN_FILENAME,
    FALSE_INVESTIGATION_FILENAME,
    FALSE_FINAL_FILENAME,
    SCHEMA_VERSION,
    atomic_write_json,
    load_json,
    sha256_file,
    abstract_screen_input_sha256,
    validate_source_model,
    SourceStateError,
    investigation_payload_sha256,
    validate_false_payload,
)
from test_arxivmath_source_scripts import FakeClient, source_fixture

MODEL = "openai/gpt-6-astra"
QUOTES = {
    "result": "Our principal theorem constructs an object satisfying H and violating C.",
    "proof": "The construction satisfies hypothesis H by Lemma 2, but its invariant violates conclusion C.",
    "prior_claim": "Earlier authors conjectured that every object satisfying H must satisfy conclusion C.",
    "prior_work": "That conjecture remained unresolved before the counterexample constructed in this article.",
    "difficulty": "The obstruction defeats all earlier constructions and requires the new mechanism of Section 3.",
}
SOURCE = "% MATHARENA_SOURCE_BEGIN file=main.tex\n" + "\n".join(QUOTES.values()) + "\n"


def candidate():
    return {
        "keep": True,
        "true_statement": "There exists an object with property H whose invariant does not satisfy C.",
        "false_statement": "Every object with property H has an invariant satisfying C.",
        "falsity_explanation": "PRIVATE_GENERATOR_REFUTATION: the witness has H and fails C.",
        "claim_kind": "disproved_conjecture",
        "prior_claim": "Earlier authors conjectured H implies C.",
        "prior_work_status": "new_refutation",
        "importance": "main",
        "basis_summary": "PRIVATE_GENERATOR_BASIS: the construction contradicts the prior conjecture.",
        "plausibility_rationale": "PRIVATE_GENERATOR_PLAUSIBILITY: earlier constructions satisfy C.",
        "difficulty_rationale": "PRIVATE_GENERATOR_DIFFICULTY: a new obstruction is required.",
        "easy_refutation_audit": "PRIVATE_GENERATOR_AUDIT: standard examples satisfy C.",
        "evidence_quotes": [{"role": role, "quote": quote} for role, quote in QUOTES.items()],
    }


def render_prompt(name, **fields):
    return Path(f"arxivmath/prompts/broken/source_{name}.md").read_text().format(**fields)


def verdict(keep=True):
    # Use the actual prompt's output example so model fixtures cannot conceal a
    # mismatch between instructions and the validator's required fields.
    prompt = render_prompt(
        "verify",
        arxiv_id="2608.00001v1",
        source_text=SOURCE,
        true_statement=candidate()["true_statement"],
        false_statement=candidate()["false_statement"],
    )
    response = extract_json(prompt.split("## Output", 1)[1])
    response.update(
        keep=keep,
        research_difficult=keep,
        reason=(
            "INDEPENDENT_REFUTATION: H holds by Lemma 2; the invariant negates C. "
            "The new main construction disproves the documented prior claim."
        ),
        evidence_quotes=[{"role": role, "quote": quote} for role, quote in QUOTES.items()],
    )
    return response


def test_revised_prompt_output_examples_match_validators():
    screen = render_prompt("screen", title="Example", abstract="Example abstract")
    assert screen_abstracts.validate_abstract_screen(extract_json(screen)) == {"decision": "accept"}
    assert '{"decision": "reject"}' in screen
    prompt = render_prompt("investigate", arxiv_id="2608.00001v1", source_text=SOURCE)
    rejection = extract_json(prompt.split("## Output", 1)[1])
    assert validate_false_payload(rejection, SOURCE)["valid"] is True
    acceptance = extract_json(prompt.split("For acceptance, return", 1)[1])
    assert set(acceptance) == set(candidate())
    acceptance.update(
        claim_kind="disproved_conjecture", importance="main", evidence_quotes=candidate()["evidence_quotes"]
    )
    assert validate_false_payload(acceptance, SOURCE)["valid"] is True
    item = validate_false_payload(candidate(), SOURCE)["investigation"]
    for keep in (True, False):
        result = validate_false_payload(verdict(keep), SOURCE, investigation=item)
        assert result["valid"] is True, result
        assert result["passed"] is keep


def forbidden_client(*args, **kwargs):
    raise AssertionError("Live model calls are forbidden in these tests")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for module in (screen_abstracts, investigate_sources, verify_questions):
        monkeypatch.setattr(module, "create_query_client", forbidden_client)


@pytest.fixture
def papers(tmp_path):
    root, paper, cache = source_fixture(tmp_path, source_text=SOURCE)
    metadata = load_json(paper / "metadata.json")
    config_path = f"configs/models/{MODEL}.yaml"
    prompt = "arxivmath/prompts/broken/source_screen.md"
    atomic_write_json(
        paper / FALSE_SCREEN_FILENAME,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "decision": "accept",
            "model_config_path": config_path,
            "model_config_sha256": sha256_file(config_path),
            "identity": {
                "metadata_input_sha256": abstract_screen_input_sha256(metadata),
                "prompt_sha256": sha256_file(prompt),
                "rendered_prompt_sha256": "fixture",
                "model_config_sha256": sha256_file(config_path),
            },
        },
    )
    return root, paper, cache


def run(monkeypatch, module, papers, *extra):
    root, _, cache = papers
    argv = [module.__name__, "--paper-root", str(root)]
    if module is not screen_abstracts:
        argv += ["--source-cache", str(cache)]
    if module is not exporter:
        argv += ["--false"]
    if module in (screen_abstracts, investigate_sources, verify_questions):
        argv += ["--model-config", MODEL]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    return module.main()


def prepare(monkeypatch, papers, *, keep=True, unlocated_quote=False):
    generated, verified = candidate(), verdict(keep)
    if unlocated_quote:
        for payload in (generated, verified):
            payload["evidence_quotes"][1]["quote"] = QUOTES["proof"].replace("Lemma 2", "Lemma~2")
    generator, verifier = FakeClient(generated), FakeClient(verified)
    monkeypatch.setattr(investigate_sources, "create_query_client", lambda *a, **k: generator)
    monkeypatch.setattr(verify_questions, "create_query_client", lambda *a, **k: verifier)
    assert run(monkeypatch, investigate_sources, papers) == 0
    assert run(monkeypatch, verify_questions, papers) == 0
    monkeypatch.setattr(investigate_sources, "create_query_client", forbidden_client)
    monkeypatch.setattr(verify_questions, "create_query_client", forbidden_client)
    return generator, verifier


def approve(papers):
    path = papers[1] / FALSE_FINAL_FILENAME
    annotation = load_json(path)
    annotation["review"] = {"status": "keep", "binding": annotation["source_first"]["review_binding"]}
    annotation["stage"] = "human_accepted"
    atomic_write_json(path, annotation)


@pytest.mark.parametrize("stage", ["screen", "investigate", "verify"])
def test_prompt_edits_allow_review_and_export(monkeypatch, papers, tmp_path, stage):
    prompts = {}
    for name in ("screen", "investigate", "verify"):
        prompts[name] = tmp_path / f"{name}.md"
        prompts[name].write_text(Path(f"arxivmath/prompts/broken/source_{name}.md").read_text())
    screen = load_json(papers[1] / FALSE_SCREEN_FILENAME)
    screen["prompt_path"] = str(prompts["screen"])
    atomic_write_json(papers[1] / FALSE_SCREEN_FILENAME, screen)
    monkeypatch.setattr(investigate_sources, "create_query_client", lambda *a, **k: FakeClient(candidate()))
    monkeypatch.setattr(verify_questions, "create_query_client", lambda *a, **k: FakeClient(verdict()))
    assert run(monkeypatch, investigate_sources, papers, "--prompt", str(prompts["investigate"])) == 0
    assert run(monkeypatch, verify_questions, papers, "--prompt", str(prompts["verify"])) == 0
    state_before = (papers[1] / FALSE_INVESTIGATION_FILENAME).read_bytes()
    prompts[stage].write_text(prompts[stage].read_text() + "\nUpdated guidance for future runs.\n")
    monkeypatch.setattr(investigate_sources, "create_query_client", forbidden_client)
    monkeypatch.setattr(verify_questions, "create_query_client", forbidden_client)

    annotation = load_json(papers[1] / FALSE_FINAL_FILENAME)
    assert annotation["stage"] == "selected_for_human_review"
    assert annotation["true_statement"] == candidate()["true_statement"]
    assert (papers[1] / FALSE_INVESTIGATION_FILENAME).read_bytes() == state_before

    approve(papers)
    report = audit_paper(papers[1], source_cache=papers[2], false_mode=True, allow_source_unavailable=True)
    assert report["ready"] is True, report["blockers"]
    out = tmp_path / "release"
    assert run(monkeypatch, exporter, papers, "--out-dir", str(out), "--date", "2026-08-01") == 0


def test_processing_stages_use_accepted_papers_with_unfinished_screening(monkeypatch, papers):
    root, accepted, _ = papers
    pending = root / "2608.00002"
    atomic_write_json(pending / "metadata.json", {"id": pending.name, "title": "Not screened yet"})
    stale = root / "2608.00003"
    atomic_write_json(stale / "metadata.json", {"id": stale.name, "title": "Changed abstract"})
    atomic_write_json(stale / FALSE_SCREEN_FILENAME, load_json(accepted / FALSE_SCREEN_FILENAME))
    partial = root / "2608.00004"
    partial.mkdir()
    (partial / "metadata.json").write_text('{"title":')

    generator, verifier = prepare(monkeypatch, papers)

    assert len(generator.queries) == len(verifier.queries) == 1
    assert load_json(accepted / FALSE_FINAL_FILENAME)["stage"] == "selected_for_human_review"
    for module in (investigate_sources, verify_questions):
        assert run(monkeypatch, module, papers) == 0  # Reuse without any new model calls.
    for paper in (pending, stale, partial):
        assert not (paper / FALSE_INVESTIGATION_FILENAME).exists()
        assert not (paper / FALSE_FINAL_FILENAME).exists()


@pytest.mark.parametrize("unlocated_quote", [False, True])
def test_full_shared_flow_uses_gpt6_and_preserves_benchmark_contract(monkeypatch, papers, tmp_path, unlocated_quote):
    _, verifier = prepare(monkeypatch, papers, unlocated_quote=unlocated_quote)
    prompt = verifier.queries[0][0]["content"]
    assert "PRIVATE_GENERATOR" not in prompt
    assert candidate()["true_statement"] in prompt and candidate()["false_statement"] in prompt
    assert QUOTES["proof"] in prompt
    state = load_json(papers[1] / FALSE_INVESTIGATION_FILENAME)
    assert state["generation"]["model"] == state["verification"]["model"] == "gpt-6-astra"
    assert not (papers[1] / "source_false_verification.json").exists()
    assert run(monkeypatch, investigate_sources, papers) == run(monkeypatch, verify_questions, papers) == 0
    assert run(monkeypatch, audit_pipeline, papers) == 1  # Human review is pending.
    approve(papers)
    assert run(monkeypatch, audit_pipeline, papers) == 0
    out = tmp_path / "release"
    assert run(monkeypatch, exporter, papers, "--out-dir", str(out), "--date", "2026-08-01") == 0
    assert (out / "problems/1.tex").read_text().strip() == candidate()["false_statement"]
    assert (out / "original/1.tex").read_text().strip() == candidate()["true_statement"]
    exported = json.loads((out / "refutations/1.json").read_text())
    assert exported["verification"]["reason"] == verdict()["reason"]
    for record in (state["investigation"], exported["verification"]):
        evidence = record["evidence"][1]
        assert ("start_char" in evidence) is not unlocated_quote
        assert evidence["quote"] == (
            QUOTES["proof"].replace("Lemma 2", "Lemma~2") if unlocated_quote else QUOTES["proof"]
        )
    assert "independent_refutation" not in exported
    assert "2608.00001v1" in (out / "source.csv").read_text()
    grading = load_json(out / "grading_scheme.json")[0]
    assert grading["ground_truth_proofs"] == []
    assert grading["points"] == 3
    with pytest.raises(SystemExit):
        run(monkeypatch, exporter, papers, "--out-dir", str(out), "--date", "2026-08-01")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(keep="true"),
        lambda p: p.update(prior_claim=None),
        lambda p: p.update(claim_kind="invented_claim"),
        lambda p: p.update(importance="secondary"),
        lambda p: p.update(prior_work_status="already_refuted"),
        lambda p: p.update(false_statement="According to this paper all objects satisfy C."),
        lambda p: p["evidence_quotes"][0].update(quote=""),
        lambda p: p["evidence_quotes"][0].update(quote=None),
        lambda p: p.update(evidence_quotes=p["evidence_quotes"][:-1]),
        lambda p: p.update(unexpected_field=True),
    ],
)
def test_invalid_items_are_rejected(mutation):
    item = candidate()
    mutation(item)
    assert validate_false_payload(item, SOURCE)["valid"] is False


@pytest.mark.parametrize("quote,source", [(QUOTES["proof"], SOURCE + QUOTES["proof"]), ("hypothesis H", SOURCE)])
def test_ambiguous_or_short_quotes_do_not_block_acceptance(quote, source):
    generated, verified = candidate(), verdict()
    for payload in (generated, verified):
        payload["evidence_quotes"][1]["quote"] = quote
    result = validate_false_payload(generated, source)
    assert result["valid"] is True, result
    item = result["investigation"]
    result = validate_false_payload(verified, source, investigation=item)
    assert result["valid"] is True and result["passed"] is True, result
    for evidence in (item["evidence"][1], result["evidence"][1]):
        assert evidence["quote"] == quote
        assert "start_char" not in evidence


def test_verifier_rejects_insufficient_difficulty_and_inconsistent_checks(monkeypatch, papers):
    prepare(monkeypatch, papers, keep=False)
    assert load_json(papers[1] / FALSE_FINAL_FILENAME)["stage"] == "automatic_verification_rejected"
    item = validate_false_payload(candidate(), SOURCE)["investigation"]
    response = verdict()
    response["hypotheses_match"] = False
    assert validate_false_payload(response, SOURCE, investigation=item)["valid"] is False
    response = verdict()
    response["research_difficult"] = "true"
    assert validate_false_payload(response, SOURCE, investigation=item)["valid"] is False


@pytest.mark.parametrize(
    "change", ["metadata", "source", "item", "review", "screen", "version", "config", "verifier_prompt"]
)
def test_shared_audit_blocks_stale_records(monkeypatch, papers, change):
    prepare(monkeypatch, papers)
    approve(papers)
    _, paper, cache = papers
    if change in ("metadata", "version"):
        metadata = load_json(paper / "metadata.json")
        metadata["abstract" if change == "metadata" else "versioned_id"] = (
            "Changed" if change == "metadata" else paper.name + "v2"
        )
        atomic_write_json(paper / "metadata.json", metadata)
    elif change == "source":
        (cache / (paper.name + "v1") / "combined_source.tex").write_text(SOURCE + "changed")
    elif change == "review":
        annotation = load_json(paper / FALSE_FINAL_FILENAME)
        annotation["false_statement"] = "An unverified edited claim."
        atomic_write_json(paper / FALSE_FINAL_FILENAME, annotation)
    elif change == "screen":
        record = load_json(paper / FALSE_SCREEN_FILENAME)
        record["identity"]["prompt_sha256"] = "changed"
        atomic_write_json(paper / FALSE_SCREEN_FILENAME, record)
    else:
        state = load_json(paper / FALSE_INVESTIGATION_FILENAME)
        if change == "item":
            state["investigation"]["false_statement"] += " With a changed hypothesis."
            state["investigation"]["investigation_sha256"] = investigation_payload_sha256(state["investigation"])
        elif change == "config":
            state["generation"]["model_config_sha256"] = "changed"
        else:
            state["verification"]["prompt_sha256"] = "changed"
        atomic_write_json(paper / FALSE_INVESTIGATION_FILENAME, state)
    assert not audit_paper(paper, source_cache=cache, false_mode=True, allow_source_unavailable=True)["ready"]


def test_unavailable_sources_are_skipped_without_an_opt_in(monkeypatch, papers):
    paper = papers[1]
    (paper / "source_ref.json").unlink()
    atomic_write_json(paper / "source_ingestion.json", {"status": "failed"})
    assert run(monkeypatch, investigate_sources, papers) == 0
    assert run(monkeypatch, verify_questions, papers) == 0
    assert run(monkeypatch, audit_pipeline, papers) == 0
    assert load_json(paper / FALSE_FINAL_FILENAME)["stage"] == "source_unavailable"


def test_reverification_requires_fresh_review(monkeypatch, papers):
    prepare(monkeypatch, papers)
    approve(papers)
    monkeypatch.setattr(verify_questions, "create_query_client", lambda *a, **k: FakeClient(verdict()))
    assert run(monkeypatch, verify_questions, papers, "--overwrite") == 0
    assert "review" not in load_json(papers[1] / FALSE_FINAL_FILENAME)


def test_verifier_can_reject_an_undocumented_claim():
    item = validate_false_payload(candidate(), SOURCE)["investigation"]
    response = verdict(False)
    response.update(natural_claim=False, evidence_quotes=[])
    result = validate_false_payload(response, SOURCE, investigation=item)
    assert result["valid"] is True and result["passed"] is False
    response.update(keep=True, **{key: True for key in FALSE_CHECKS})
    assert validate_false_payload(response, SOURCE, investigation=item)["valid"] is False


def test_non_gpt6_model_configs_are_refused():
    with pytest.raises(SourceStateError, match="GPT-6"):
        validate_source_model({"harness": "codex", "model": "another-model"}, false_mode=True)


@pytest.mark.parametrize("unlocated_quote", [False, True])
def test_review_recovers_saved_results_and_is_read_only(monkeypatch, papers, unlocated_quote):
    prepare(monkeypatch, papers, unlocated_quote=unlocated_quote)
    root, paper, _ = papers
    # Opening the GUI must suffice even if a run stopped before writing its review file.
    (paper / FALSE_FINAL_FILENAME).unlink()
    state_before = (paper / FALSE_INVESTIGATION_FILENAME).read_bytes()
    for name, value in {
        "PAPER_ROOT": str(root),
        "ANNOTATION_FILENAME_OVERRIDE": None,
        "CHECK_ONLY_KEPT": False,
        "FALSE_MODE": True,
        "LEAN_MODE": False,
    }.items():
        monkeypatch.setattr(review_app, name, value)
    review_app.app.config.update(TESTING=True, SECRET_KEY="test")
    client = review_app.app.test_client()
    html = client.get(f"/paper/{paper.name}").get_data(as_text=True)
    assert (paper / FALSE_INVESTIGATION_FILENAME).read_bytes() == state_before
    assert "INDEPENDENT_REFUTATION" in html
    assert ("Source location unavailable" in html) is unlocated_quote
    if unlocated_quote:
        assert QUOTES["proof"].replace("Lemma 2", "Lemma~2") in html
    assert "Independent easy-refutation audit:" not in html
    assert '<textarea id="false_statement"' not in html
    annotation = load_json(paper / FALSE_FINAL_FILENAME)
    assert "perturbed_statement" not in annotation and "original_statement" not in annotation
    assert (
        client.post(
            f"/paper/{paper.name}/annotate",
            data={
                "status": "keep",
                "review_binding": annotation["source_first"]["review_binding"],
                "false_statement": "ignored edit",
            },
        ).status_code
        == 302
    )
    assert run(monkeypatch, audit_pipeline, papers) == 0
    assert (
        client.post(
            f"/paper/{paper.name}/annotate",
            data={
                "status": "keep",
                "review_binding": "stale page",
            },
        ).status_code
        == 409
    )


def test_default_shell_runs_only_screening(tmp_path):
    executable, log = tmp_path / "uv", tmp_path / "calls"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$BROKENARXIV_TEST_LOG"\n')
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("ARXIV_FALSE_")}
    env.update(PATH=f"{tmp_path}:{env['PATH']}", BROKENARXIV_TEST_LOG=str(log))
    subprocess.run(["bash", "arxivmath/scripts/create_false.sh"], env=env, check=True, capture_output=True)
    calls = log.read_text().splitlines()
    assert len(calls) == 1 and "screen_abstracts.py --false" in calls[0]
    assert "--model-config" not in calls[0]  # Use the stage's own default.



@pytest.mark.parametrize(
    "source_status,verification_status",
    [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (130, 0)],
)
def test_prepare_verifies_completed_candidates_after_source_failure(
    tmp_path, source_status, verification_status
):
    executable, log = tmp_path / "uv", tmp_path / "calls"
    executable.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$BROKENARXIV_TEST_LOG"\n'
        'case "$*" in\n'
        f'  *investigate_sources.py*) exit {source_status} ;;\n'
        f'  *verify_questions.py*) exit {verification_status} ;;\n'
        'esac\n'
    )
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("ARXIV_FALSE_")}
    env.update(PATH=f"{tmp_path}:{env['PATH']}", BROKENARXIV_TEST_LOG=str(log))
    result = subprocess.run(
        ["bash", "arxivmath/scripts/create_false.sh", "prepare"],
        env=env, capture_output=True, text=True,
    )
    calls = log.read_text().splitlines()
    if source_status not in (0, 1):
        assert len(calls) == 3
        assert result.returncode == source_status
        return
    assert len(calls) == 4
    assert "verify_questions.py --false" in calls[-1]
    assert result.returncode == (verification_status or source_status)
    assert "Review: uv run python arxivmath/app.py --false" in result.stdout
    if source_status:
        assert "continuing verification of completed candidates" in result.stderr

def test_oversized_sources_never_call_codex(monkeypatch, tmp_path):
    root, paper, cache = source_fixture(tmp_path, source_text=SOURCE + ("x" * 50_000))
    metadata = load_json(paper / "metadata.json")
    config_path = f"configs/models/{MODEL}.yaml"
    atomic_write_json(
        paper / FALSE_SCREEN_FILENAME,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "decision": "accept",
            "model_config_path": config_path,
            "model_config_sha256": sha256_file(config_path),
            "identity": {
                "metadata_input_sha256": abstract_screen_input_sha256(metadata),
                "prompt_sha256": sha256_file("arxivmath/prompts/broken/source_screen.md"),
                "rendered_prompt_sha256": "fixture",
                "model_config_sha256": sha256_file(config_path),
            },
        },
    )
    papers = root, paper, cache
    assert run(monkeypatch, investigate_sources, papers, "--max-source-tokens", "20000") == 0
    assert run(monkeypatch, verify_questions, papers) == 0
    assert load_json(paper / FALSE_FINAL_FILENAME)["stage"] == "source_oversized"


def test_export_selects_reviewed_items_while_other_papers_are_unfinished(monkeypatch, papers, tmp_path):
    prepare(monkeypatch, papers)
    approve(papers)
    root, _, _ = papers
    for paper_id, annotation in (
        ("2608.00002", {}),
        ("2608.00003", {"keep": True, "stage": "selected_for_human_review"}),
        ("2608.00004", {"keep": False, "stage": "awaiting_verification"}),
        ("2608.00005", {"keep": False, "stage": "human_rejected"}),
    ):
        paper = root / paper_id
        atomic_write_json(paper / "metadata.json", {"title": "Unfinished or rejected"})
        atomic_write_json(paper / FALSE_FINAL_FILENAME, annotation)
    (root / "2608.00002" / "metadata.json").write_text('{"title":')
    out = tmp_path / "release"
    assert run(monkeypatch, exporter, papers, "--out-dir", str(out), "--date", "2026-08-01") == 0
    assert len(list((out / "problems").glob("*.tex"))) == 1
    manifest = load_json(out / "generation_manifest.json")
    assert manifest["exported_papers"] == 1
    assert [item["paper_id"] for item in manifest["items"]] == [papers[1].name]


def test_export_still_refuses_a_stale_accepted_item(monkeypatch, papers, tmp_path):
    prepare(monkeypatch, papers)
    approve(papers)
    annotation = load_json(papers[1] / FALSE_FINAL_FILENAME)
    annotation["false_statement"] += " With a changed assumption."
    atomic_write_json(papers[1] / FALSE_FINAL_FILENAME, annotation)
    out = tmp_path / "release"
    assert run(monkeypatch, exporter, papers, "--out-dir", str(out), "--date", "2026-08-01") == 1
    assert not out.exists()


@pytest.mark.parametrize("points, normalized", [(0, 0), (1, 1 / 3), (2, 2 / 3), (3, 1), (4, None), (None, None)])
def test_gemini_flash_judge_uses_antigravity_and_three_point_scale(monkeypatch, tmp_path, points, normalized):
    from copy import deepcopy
    from types import SimpleNamespace
    import yaml
    from harness_wrapper import AgentEvent, TokenUsage
    from matharena.solvers.harness_solver import HarnessSolver
    from matharena.solvers.judges.simple_judge import SimpleJudge
    from scripts.judge.judge import build_judgment_entry, recompute_correct_and_pass_at_1

    full_config = yaml.safe_load(Path("configs/judges/arxiv_judge_gemini_38_flash.yaml").read_text())
    config = yaml.safe_load(Path(f"configs/{full_config['scaffold_config']}.yaml").read_text())
    config["model_config"] = yaml.safe_load(Path(f"configs/models/{full_config['model_config']}.yaml").read_text())
    config.update(full_config["override"])
    original = deepcopy(config)
    assert full_config["judge_points_max"] == 3
    assert config["n_threads"] > 0
    requests, agents = [], []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.root = kwargs["dir"]
            self.session_id = "judge-session"
            self.model_request_log_path = self.root / "model_requests.jsonl"
            agents.append(self)

        def run(self, prompt):
            requests.append(prompt)
            self.model_request_log_path.write_text(prompt)
            if points is None:
                raise RuntimeError("Simulated incomplete CLI turn")
            return [AgentEvent(type="message", role="assistant", content=(
                f"<points>{points}</points><assessment>Mock assessment.</assessment>"
            ))]

        def get_tokens(self):
            return TokenUsage(input_tokens=10, output_tokens=5)

    def no_legacy_api(**kwargs):
        pytest.fail("Harness judging must not start the legacy API/Modal tool loop")

    monkeypatch.setenv("MATHARENA_REQUEST_LOG_DIR", str(tmp_path))
    monkeypatch.setattr("matharena.solvers.judges.simple_judge.APIClient", no_legacy_api)
    monkeypatch.setattr(HarnessSolver, "_prepare_harness_cli", lambda self: None)
    monkeypatch.setattr(HarnessSolver, "_build_model", lambda self: "offline-model")
    monkeypatch.setattr(HarnessSolver, "_build_sandbox", lambda self, path: SimpleNamespace(root=path))
    monkeypatch.setattr(HarnessSolver, "_agent_executable", lambda self: None)
    monkeypatch.setattr(HarnessSolver, "_agent_environment", lambda self: {})
    monkeypatch.setattr("matharena.solvers.harness_solver.Agent", FakeAgent)
    judge = SimpleJudge(7, 11, 2, config)
    solver = judge.harness_solver
    assert solver.config["model"] == "gemini-3.8-flash"
    assert solver.config["reasoning_effort"] == "high"
    assert solver.harness_version == config["model_config"]["harness_version"]
    assert solver.harness_config["auth"] == "api"
    assert "tools" not in solver.config and "max_tool_calls" not in solver.config
    retry = SimpleJudge(7, 11, 2, config)
    assert solver._workspace_for(11, 2) != retry.harness_solver._workspace_for(11, 2)
    response = judge.solve("False assertion", "", [], "This is false.", "True negation")
    assert response.idx == 7 and response.points == points
    assert config == original
    assert agents[0].options["type"] == "gravity"
    assert agents[0].options["tools_enabled"] is True
    assert agents[0].options["auto_fallback"] is False
    assert agents[0].root == solver._workspace_for(11, 2)
    assert solver._response_agents == {}
    prompt = requests[0]
    assert "False assertion" in prompt and "True negation" in prompt and "This is false." in prompt
    if points is not None:
        assert response.detailed_cost["input_tokens"] == 10
        assert response.detailed_cost["output_tokens"] == 5
        assert response.history[0]["harness"] == "gravity"
        assert response.history[0]["session_id"] == "judge-session"
        assert response.history[0]["model_requests"]

    entry = build_judgment_entry(response, 3, "", "judges/arxiv_judge_gemini_38_flash", full_config["judge_points_max"])
    if normalized is None:
        assert entry is None
        return
    assert entry["points"] == points
    assert entry["max_points"] == entry["details"][0]["max_points"] == 3
    assert entry["history"] == response.history
    assert entry["cost"] == response.detailed_cost
    run_data = {"messages": [response.history[0]["messages"]], "judgment": [[entry]]}
    recompute_correct_and_pass_at_1(run_data)
    assert run_data["correct"] == pytest.approx([normalized])
    assert run_data["pass_at_1"] == pytest.approx(normalized)


@pytest.mark.parametrize("false_mode", [False, True])
@pytest.mark.parametrize("batch_size", [7, 256])
def test_screening_paces_batches_and_resumes_without_repeating_completed_work(monkeypatch, tmp_path, batch_size, false_mode):
    root = tmp_path / "papers"
    paper_ids = [f"2608.{index:05d}" for index in range(61)]
    for paper_id in paper_ids:
        atomic_write_json(root / paper_id / "metadata.json", {
            "id": paper_id, "title": paper_id, "abstract": "We disprove a longstanding conjecture.",
        })
    now = 0
    pauses = []
    starts = []

    def sleep(seconds):
        nonlocal now
        pauses.append(seconds)
        now += seconds

    class Client:
        def run_queries(self, queries):
            nonlocal now
            offset = len(starts)
            starts.extend([now] * len(queries))
            # Vary execution duration and return results out of order to check
            # that pacing preserves the mapping from batch indices to papers.
            now += offset % 9
            for index in reversed(range(len(queries))):
                decision = "accept" if (offset + index) % 2 == 0 else "reject"
                yield index, [{"role": "assistant", "content": json.dumps({"decision": decision})}], {"cost": 0}

    monkeypatch.setattr(screen_abstracts.time, "sleep", sleep)
    monkeypatch.setattr(screen_abstracts, "create_query_client", lambda _, **kwargs: Client())
    monkeypatch.setattr(sys, "argv", [
        "screen_abstracts.py", "--paper-root", str(root), "--batch-size", str(batch_size),
    ] + (["--false"] if false_mode else []))
    assert screen_abstracts.main() == 0
    assert len(starts) == len(paper_ids)
    assert pauses
    for start in starts:
        assert sum(start <= other < start + 60 for other in starts) <= 50
    screen_filename = FALSE_SCREEN_FILENAME if false_mode else "abstract_screen.json"
    for index, paper_id in enumerate(paper_ids):
        record = load_json(root / paper_id / screen_filename)
        assert record["decision"] == ("accept" if index % 2 == 0 else "reject")
    assert len(pauses) == (len(paper_ids) - 1) // min(batch_size, 25)

    pauses.clear()
    assert screen_abstracts.main() == 0
    assert len(starts) == len(paper_ids)
    assert pauses == []


def test_screening_does_not_pause_without_a_following_batch(monkeypatch):
    def forbidden_sleep(_):
        raise AssertionError("No pacing expected")

    monkeypatch.setattr(screen_abstracts.time, "sleep", forbidden_sleep)
    queries = list(range(25))
    assert list(screen_abstracts.screening_batches(queries, 256)) == [(0, queries)]
    assert list(screen_abstracts.screening_batches([], 256)) == []
