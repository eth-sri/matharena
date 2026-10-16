import json
import sys
from pathlib import Path

import pytest

from arxivmath.scripts.arxiv.export_accepted_questions import is_accepted
from arxivmath.scripts.source import (
    download_sources,
    investigate_sources,
    screen_abstracts,
)
from arxivmath.scripts.source import verify_questions
from matharena.arxivmath_audit import audit_paper
from matharena.arxivmath_source import (
    SCHEMA_VERSION,
    final_annotation,
    SourceStateError,
    abstract_screen_input_sha256,
    build_investigation_source,
    investigation_payload_sha256,
    read_source_artifacts,
    sha256_file,
    sha256_text,
)


SOURCE_QUOTE = "The principal calculation proves that the invariant N has exact value seven."
SOURCE_TEXT = "% MATHARENA_SOURCE_BEGIN file=main.tex\n" "\\section{Main results}\n" f"{SOURCE_QUOTE}\n"


def basis_summary():
    return (
        "The question uses the principal theorem in the main-results section, together with the "
        "displayed calculation that evaluates the invariant. The cited passage supplies the exact "
        "hypotheses and normalization needed by the benchmark statement, while the requested scalar "
        "is a substantive conclusion of the article rather than background material."
    )


def accepted_response():
    return {
        "keep": True,
        "question": "Under the stated hypotheses, what is the exact value of the invariant N?",
        "answer": "7",
        "answer_type": "exact_scalar",
        "declared_variables": [],
        "basis_summary": basis_summary(),
        "novelty_type": "new_exact_value",
        "importance": "main",
        "refutation_status": "not_applicable",
        "prior_claim": None,
        "new_result": None,
        "evidence_quotes": [SOURCE_QUOTE],
    }


def verification_response(answer="7"):
    return {
        "keep": True,
        "source_supported": True,
        "self_contained": True,
        "unique_and_well_defined": True,
        "answer_type_supported": True,
        "no_missing_context": True,
        "no_answer_leak": True,
        "research_substantive": True,
        "novelty_supported": True,
        "refutation_supported": True,
        "derived_answer": answer,
        "reason": "The source evidence uniquely determines the requested value.",
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def source_fixture(tmp_path, paper_id="2608.00001", source_text=SOURCE_TEXT):
    paper_root = tmp_path / "paper"
    paper = paper_root / paper_id
    metadata = {
        "id": paper_id,
        "versioned_id": f"{paper_id}v1",
        "title": "An exact invariant",
        "abstract": "We determine a new exact invariant.",
    }
    write_json(paper / "metadata.json", metadata)
    screen = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "decision": "accept",
        "identity": {
            "metadata_input_sha256": abstract_screen_input_sha256(metadata),
            "prompt_sha256": "screen-prompt",
            "rendered_prompt_sha256": "screen-rendered",
            "model_config_sha256": "screen-config",
        },
    }
    write_json(paper / "abstract_screen.json", screen)

    cache_root = tmp_path / "source-cache"
    cache_key = f"{paper_id}v1"
    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True)
    (cache_dir / "combined_source.tex").write_text(source_text, encoding="utf-8")
    source_sha = sha256_text(source_text)
    manifest = {
        "schema_version": 2,
        "arxiv_id": f"{paper_id}v1",
        "tex": {"combined_sha256": source_sha, "unresolved_includes": []},
    }
    write_json(cache_dir / "source_manifest.json", manifest)
    reference = {
        "schema_version": SCHEMA_VERSION,
        "arxiv_id": f"{paper_id}v1",
        "base_id": paper_id,
        "version": 1,
        "cache_key": cache_key,
        "manifest_sha256": sha256_file(cache_dir / "source_manifest.json"),
        "combined_source_sha256": source_sha,
        "combined_source_bytes": len(source_text.encode()),
        "unresolved_include_count": 0,
        "unresolved_includes": [],
        "unresolved_includes_allowed": False,
    }
    write_json(paper / "source_ref.json", reference)
    write_json(paper / "source_ingestion.json", {"status": "completed"})
    return paper_root, paper, cache_root


def test_read_source_artifacts_preserves_crlf_bytes(tmp_path):
    source_text = SOURCE_TEXT.replace("\n", "\r\n")
    _paper_root, paper, cache_root = source_fixture(tmp_path, source_text=source_text)

    loaded, _provenance, _cache_dir = read_source_artifacts(paper, cache_root)

    assert loaded == source_text


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.queries = []

    def run_queries(self, queries):
        self.queries.extend(queries)
        for index, _ in enumerate(queries):
            yield index, [{"role": "assistant", "content": json.dumps(self.response)}], {"cost": 1.25}


def patch_model(monkeypatch, module, tmp_path, response, name):
    config = tmp_path / f"{name}.yaml"
    config.write_text(f"model: {name}\n", encoding="utf-8")
    client = FakeClient(response)
    monkeypatch.setattr(module, "resolve_model_config_path", lambda _: str(config))
    monkeypatch.setattr(module, "load_model_config", lambda _: {"model": name})
    monkeypatch.setattr(module, "create_query_client", lambda *_args, **_kwargs: client)
    return client


def test_abstract_screen_accepts_only_binary_decisions():
    assert screen_abstracts.validate_abstract_screen({"decision": "accept"}) == {"decision": "accept"}
    assert screen_abstracts.validate_abstract_screen({"decision": "reject"}) == {"decision": "reject"}
    assert not hasattr(screen_abstracts, "deterministic_refutation_triggers")

    with pytest.raises(SourceStateError, match="must contain only decision"):
        screen_abstracts.validate_abstract_screen({"decision": "accept", "reason": "extra model output"})
    with pytest.raises(SourceStateError, match="invalid decision"):
        screen_abstracts.validate_abstract_screen({"decision": "Accept"})


def test_abstract_screen_writes_binary_decision_without_generation(monkeypatch, tmp_path):
    paper_root = tmp_path / "paper"
    paper = paper_root / "2608.00001"
    write_json(
        paper / "metadata.json",
        {
            "id": "2608.00001",
            "title": "An exact invariant",
            "abstract": "We determine a new exact invariant.",
        },
    )
    client = patch_model(
        monkeypatch,
        screen_abstracts,
        tmp_path,
        {"decision": "accept"},
        "screen",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_abstracts.py",
            "--model-config",
            "screen",
            "--paper-root",
            str(paper_root),
        ],
    )

    assert screen_abstracts.main() == 0
    assert len(client.queries) == 1
    record = json.loads((paper / "abstract_screen.json").read_text())
    assert record["decision"] == "accept"
    assert not {"question", "answer", "priority", "signals"} & set(record)


@pytest.mark.parametrize("false_mode", [False, True])
def test_source_download_uses_ready_acceptances_while_other_papers_are_unfinished(monkeypatch, tmp_path, false_mode):
    root, accepted, cache = source_fixture(tmp_path)
    filename = "abstract_screen_false.json" if false_mode else "abstract_screen.json"
    if false_mode:
        (accepted / "abstract_screen.json").rename(accepted / filename)
    template = json.loads((accepted / filename).read_text())
    states = ["reject", "missing", "failed", "stale", "partial_metadata", "partial_screen"]
    for number, state in enumerate(states, start=2):
        paper = root / f"2608.{number:05d}"
        metadata = {"id": paper.name, "title": state, "abstract": "An unfinished paper."}
        write_json(paper / "metadata.json", metadata)
        record = json.loads(json.dumps(template))
        record["identity"]["metadata_input_sha256"] = abstract_screen_input_sha256(metadata)
        if state == "missing":
            continue
        if state == "reject":
            record["decision"] = "reject"
        elif state == "failed":
            record["status"] = "failed"
        elif state == "stale":
            record["identity"]["metadata_input_sha256"] = "old abstract"
        write_json(paper / filename, record)
        if state == "partial_metadata":
            (paper / "metadata.json").write_text('{"title":')
        elif state == "partial_screen":
            (paper / filename).write_text('{"decision":')

    requested = []

    def prepare(version, directory, **kwargs):
        requested.append(version.canonical_id)
        return json.loads((directory / "source_manifest.json").read_text())

    monkeypatch.setattr(download_sources, "prepare_arxiv_source", prepare)
    monkeypatch.setattr(
        download_sources, "resolve_latest_versions", lambda *a, **k: pytest.fail("Unexpected network request")
    )
    args = ["download_sources.py", "--paper-root", str(root), "--source-cache", str(cache)]
    monkeypatch.setattr(sys, "argv", args + (["--false"] if false_mode else []))

    assert download_sources.main() == 0
    assert requested == [accepted.name + "v1"]
    assert read_source_artifacts(accepted, cache)[0] == SOURCE_TEXT
    for paper in root.iterdir():
        if paper != accepted:
            assert not (paper / "source_ingestion.json").exists()
            assert not (paper / "source_ref.json").exists()


@pytest.mark.parametrize("false_mode", [False, True])
def test_source_download_with_no_ready_acceptances_is_a_noop(monkeypatch, tmp_path, false_mode):
    root, paper, cache = source_fixture(tmp_path)
    (paper / "abstract_screen.json").unlink()
    for name in ("prepare_arxiv_source", "pin_missing_metadata_versions"):
        monkeypatch.setattr(download_sources, name, lambda *a, **k: pytest.fail("Nothing should be downloaded"))
    args = ["download_sources.py", "--paper-root", str(root), "--source-cache", str(cache)]
    monkeypatch.setattr(sys, "argv", args + (["--false"] if false_mode else []))
    assert download_sources.main() == 0


def test_arxiv_atom_version_resolution_requires_every_exact_revision():
    payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2608.00001v2</id></entry>
  <entry><id>https://export.arxiv.org/abs/2608.00002v7</id></entry>
</feed>
"""
    requested_urls = []
    resolved = download_sources.resolve_latest_versions(
        ["2608.00002", "2608.00001"],
        api_url="https://export.arxiv.org/api/query",
        user_agent="test",
        batch_size=100,
        timeout=1,
        retries=0,
        request_interval=0,
        fetch_xml=lambda url: requested_urls.append(url) or payload,
    )
    assert resolved == {"2608.00001": "2608.00001v2", "2608.00002": "2608.00002v7"}
    assert len(requested_urls) == 1
    assert "id_list=2608.00001%2C2608.00002" in requested_urls[0]

    missing_payload = b"""<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2608.00001v2</id></entry>
</feed>"""
    with pytest.raises(SourceStateError, match="omitted 1 requested ids"):
        download_sources.parse_version_feed(missing_payload, {"2608.00001", "2608.00003"})


def test_investigator_makes_one_call_and_writes_one_question(monkeypatch, tmp_path):
    paper_root, paper, cache_root = source_fixture(tmp_path)
    client = patch_model(monkeypatch, investigate_sources, tmp_path, accepted_response(), "investigator")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "investigate_sources.py",
            "--model-config",
            "investigator",
            "--paper-root",
            str(paper_root),
            "--source-cache",
            str(cache_root),
            "--max-source-tokens",
            "20000",
        ],
    )
    assert investigate_sources.main() == 0
    assert len(client.queries) == 1
    prompt = client.queries[0][0]["content"]
    assert "Abstract-screen record" not in prompt
    assert "We determine a new exact invariant." not in prompt
    state = json.loads((paper / "source_investigation.json").read_text())
    assert state["status"] == "completed"
    assert state["abstract_screen"] == {"decision": "accept"}
    assert "screen" not in state
    assert state["investigation"]["question"] == accepted_response()["question"]
    assert "results" not in state
    assert "candidates" not in state
    assert "hardness" not in state


def test_oversized_source_is_excluded_without_model_call(monkeypatch, tmp_path):
    oversized_source = (
        "% MATHARENA_SOURCE_BEGIN file=main.tex\n"
        + ("substantive source material " * 1800)
        + "\n% MATHARENA_SOURCE_END file=main.tex\n"
    )
    paper_root, paper, cache_root = source_fixture(tmp_path, source_text=oversized_source)
    investigator = patch_model(monkeypatch, investigate_sources, tmp_path, accepted_response(), "investigator")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "investigate_sources.py",
            "--model-config",
            "investigator",
            "--paper-root",
            str(paper_root),
            "--source-cache",
            str(cache_root),
            "--max-source-tokens",
            "20000",
        ],
    )
    assert investigate_sources.main() == 0
    assert investigator.queries == []
    state = json.loads((paper / "source_investigation.json").read_text())
    assert state["status"] == "excluded"
    assert state["reason"] == "source_too_large"
    assert state["source_input"]["complete_source"] is True
    assert state["source_input"]["approximate_tokens"] > 20_000

    verifier = patch_model(monkeypatch, verify_questions, tmp_path, verification_response(), "verifier")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_questions.py",
            "--model-config",
            "verifier",
            "--paper-root",
            str(paper_root),
            "--source-cache",
            str(cache_root),
        ],
    )
    assert verify_questions.main() == 0
    assert verifier.queries == []

    final = json.loads((paper / "llm_annotation.json").read_text())
    assert final["keep"] is False
    assert final["stage"] == "source_oversized"
    report = audit_paper(paper, source_cache=cache_root, allow_source_unavailable=False)
    assert report["ready"] is True, report


def test_same_config_verifier_derives_answer_and_immediately_materializes_review_item(monkeypatch, tmp_path):
    paper_root, paper, cache_root = source_fixture(tmp_path)
    patch_model(monkeypatch, investigate_sources, tmp_path, accepted_response(), "investigator")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "investigate_sources.py",
            "--model-config",
            "investigator",
            "--paper-root",
            str(paper_root),
            "--source-cache",
            str(cache_root),
            "--max-source-tokens",
            "20000",
        ],
    )
    assert investigate_sources.main() == 0

    client = patch_model(monkeypatch, verify_questions, tmp_path, verification_response(), "investigator")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_questions.py",
            "--model-config",
            "investigator",
            "--paper-root",
            str(paper_root),
            "--source-cache",
            str(cache_root),
        ],
    )
    assert verify_questions.main() == 0
    assert len(client.queries) == 1
    verifier_prompt = client.queries[0][0]["content"]
    assert "# Gold answer" not in verifier_prompt
    assert '"answer": "7"' not in verifier_prompt

    final_path = paper / "llm_annotation.json"
    final = json.loads(final_path.read_text())
    assert final["stage"] == "selected_for_human_review"
    assert final["basis_summary"] == basis_summary()
    assert final["source_first"]["verification"]["status"] == "passed"

    blocked = audit_paper(paper, source_cache=cache_root, allow_source_unavailable=False)
    assert blocked["ready"] is False
    assert "selected_question_human_review_missing" in blocked["blockers"]

    final["review"] = {"status": "keep", "updated_at": "2026-08-30T00:00:00Z"}
    final["keep"] = True
    final["stage"] = "human_accepted"
    write_json(final_path, final)
    report = audit_paper(paper, source_cache=cache_root, allow_source_unavailable=False)
    assert report["ready"] is True, report
    assert is_accepted(final)

    from arxivmath.scripts.arxiv import export_accepted_questions

    out = tmp_path / "release"
    monkeypatch.setattr(
        sys,
        "argv",
        ["export", "--paper-root", str(paper_root), "--source-cache", str(cache_root), "--out-dir", str(out)],
    )
    assert export_accepted_questions.main() == 0
    assert (out / "problems/1.tex").read_text().strip() == accepted_response()["question"]

    final["source_first"]["basis_summary"] = "tampered"
    write_json(final_path, final)
    report = audit_paper(paper, source_cache=cache_root, allow_source_unavailable=False)
    assert report["ready"] is False
    assert "source_first_final_basis_summary_stale" in report["blockers"]


def test_verification_rejection_cannot_be_exported(tmp_path):
    investigation = {
        "keep": True,
        "question": "What is N?",
        "answer": "7",
        "answer_type": "exact_scalar",
        "declared_variables": [],
        "basis_summary": basis_summary(),
        "novelty_type": "new_exact_value",
        "importance": "main",
        "refutation_status": "not_applicable",
        "prior_claim": None,
        "new_result": None,
        "evidence": [],
    }
    state = {
        "identity": {"source_sha256": "source"},
        "source_input": {},
        "investigation": investigation,
        "verification": {
            "status": "rejected",
            "passed": False,
            "investigation_sha256": investigation_payload_sha256(investigation),
        },
    }
    final = final_annotation(state, state_sha256="state", existing={})
    assert final["stage"] == "automatic_verification_rejected"
    assert final["keep"] is False
    assert not is_accepted(final)


def test_verifier_rejects_internally_inconsistent_model_decision():
    response = verification_response()
    response["keep"] = False
    result = verify_questions.validate_verification(response, "7")
    assert result["valid"] is False
    assert result["reason"] == "keep_false_without_failed_check"


def test_rejected_investigation_is_not_a_review_candidate():
    investigation = {
        "keep": False,
        "basis_summary": basis_summary(),
        "refutation_status": "not_applicable",
        "rejection_reason": "No exact benchmark question is available.",
    }
    investigation["investigation_sha256"] = investigation_payload_sha256(investigation)
    state = {
        "identity": {"source_sha256": "source"},
        "source_input": {},
        "investigation": investigation,
    }
    final = final_annotation(state, state_sha256="state", existing={})
    assert final["stage"] == "no_question"
    assert final["keep"] is False


def test_wrapper_contains_no_discarded_pipeline_stages():
    wrapper = Path("arxivmath/scripts/create.sh").read_text()
    assert "ARXIVMATH_SCREEN_CONFIG:-openai/gpt-6-astra" in wrapper
    assert "ARXIVMATH_INVESTIGATOR_CONFIG:-openai/gpt-6-astra" in wrapper
    assert "ARXIVMATH_VERIFIER_CONFIG:-openai/gpt-6-astra" in wrapper
    assert "gemini/gemini-31-pro" not in wrapper
    for discarded in (
        "inventory_results.py",
        "generate_candidates.py",
        "score_hardness.py",
        "select_candidates.py",
        "review_eligibility.py",
    ):
        assert discarded not in wrapper
    assert "investigate_sources.py" in wrapper
    assert "verify_questions.py" in wrapper
    assert "MATHARENA_REQUEST_LOG_DIR:-${SOURCE_CACHE%/}/request-logs" in wrapper


def test_prompt_templates_format_with_literal_json_braces():
    screen = Path("arxivmath/prompts/source/abstract_screen.md").read_text()
    investigate = Path("arxivmath/prompts/source/investigate_source.md").read_text()
    verify = Path("arxivmath/prompts/source/verify_question.md").read_text()
    rendered_screen = screen.format(title="title", abstract="abstract")
    assert '{"decision":"accept"}' in rendered_screen
    assert '{"decision":"reject"}' in rendered_screen
    assert "The full source plausibly supplies all definitions" not in screen
    assert "advanced research-level mathematics" in investigate
    assert "ending in ?" not in investigate
    assert "at least 40 characters" not in investigate
    assert "uncontaminated performance" in investigate
    assert "advanced research-level mathematics" in verify
    assert "uncontaminated performance" in verify
    investigate.format(arxiv_id="x", source_text="source")
    verify.format(
        arxiv_id="x",
        question="What is N?",
        answer_type="exact_scalar",
        declared_variables="[]",
        basis_summary=basis_summary(),
        novelty_record="{}",
        evidence_packet=SOURCE_TEXT,
    )


def test_cleaned_source_manifest_hash_is_stable():
    first = build_investigation_source(SOURCE_TEXT).manifest_record()
    second = build_investigation_source(SOURCE_TEXT).manifest_record()
    assert first == second


def test_completed_items_are_reviewable_before_verification_batch_finishes(monkeypatch, tmp_path):
    root, first, cache = source_fixture(tmp_path)
    _, second, _ = source_fixture(tmp_path, paper_id="2608.00002")
    patch_model(monkeypatch, investigate_sources, tmp_path, accepted_response(), "investigator")
    argv = ["stage", "--model-config", "investigator", "--paper-root", str(root), "--source-cache", str(cache)]
    monkeypatch.setattr(sys, "argv", argv)
    assert investigate_sources.main() == 0
    patch_model(monkeypatch, verify_questions, tmp_path, verification_response(), "verifier")
    visible_during_batch = []

    class InterruptedClient:
        def run_queries(self, queries):
            assert len(queries) == 2
            yield 0, [{"role": "assistant", "content": json.dumps(verification_response())}], {"cost": 1.25}
            visible_during_batch.append(json.loads((first / "llm_annotation.json").read_text())["stage"])
            raise RuntimeError("Interrupted before the second response")

    monkeypatch.setattr(verify_questions, "create_query_client", lambda _, **kwargs: InterruptedClient())
    assert verify_questions.main() == 1
    assert visible_during_batch == ["selected_for_human_review"]
    assert json.loads((second / "llm_annotation.json").read_text())["keep"] is False

    # Continuing verification preserves the completed human decision and requests only the unfinished item.
    final_path = first / "llm_annotation.json"
    final = json.loads(final_path.read_text())
    final.update(review={"status": "keep"}, stage="human_accepted")
    write_json(final_path, final)
    reviewed_bytes = final_path.read_bytes()
    client = FakeClient(verification_response())
    monkeypatch.setattr(verify_questions, "create_query_client", lambda _, **kwargs: client)
    assert verify_questions.main() == 0
    assert len(client.queries) == 1
    assert final_path.read_bytes() == reviewed_bytes
    assert json.loads((second / "llm_annotation.json").read_text())["stage"] == "selected_for_human_review"


def test_export_cannot_use_old_annotations_without_source_records(monkeypatch, tmp_path):
    from arxivmath.scripts.arxiv import export_accepted_questions

    root = tmp_path / "papers"
    paper = root / "2608.00001"
    write_json(paper / "metadata.json", {"title": "Old abstract extraction"})
    annotation = {
        "keep": True,
        "question": "Old question",
        "answer": "7",
        "verification": {"keep": True},
        "review": {"status": "keep", "question": "Edited old question", "answer": "7"},
    }
    write_json(paper / "llm_annotation.json", annotation)
    out = tmp_path / "release"
    monkeypatch.setattr(sys, "argv", ["export", "--paper-root", str(root), "--out-dir", str(out)])
    assert not is_accepted(annotation)
    assert export_accepted_questions.main() == 1
    assert not out.exists()
