"""The output viewer must tolerate new runs that have not been judged yet."""

import copy
import importlib.util
import sys
import re
from html import unescape
from urllib.parse import parse_qs, urlsplit
from pathlib import Path

import pytest
import yaml

import matharena.configs
from matharena.json_zst import dump_json_zst, load_json_zst


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    comp_dir = tmp_path / "competitions/test"
    comp_dir.mkdir(parents=True)
    (comp_dir / "partial.yaml").write_text(
        yaml.safe_dump({"dataset_path": str(tmp_path / "data")})
    )
    monkeypatch.setattr(
        matharena.configs, "extract_existing_configs", lambda *args, **kwargs: ({}, {})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.py",
            "--comp",
            "test/partial",
            "--output-folder",
            str(tmp_path / "outputs"),
            "--competition-config-folder",
            str(comp_dir.parent),
            "--disable-overwrite",
            "--cache-dir",
            str(tmp_path / "viewer-cache"),
        ],
    )
    source = Path(__file__).resolve().parents[1] / "app/app.py"
    spec = importlib.util.spec_from_file_location(
        "matharena_test_output_viewer", source
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.app.config["TESTING"] = True
    return module


def judgment(points=0):
    return {
        "judge_id": "judges/answer_judge",
        "points": points,
        "max_points": 1,
        "details": [
            {"points": points, "max_points": 1, "desc": "First run was judged."}
        ],
        "error": None,
    }


@pytest.mark.parametrize("gold_answer", ["0", None])
@pytest.mark.parametrize("estimated", [False, True])
def test_problem_page_loads_with_short_judgment_row(viewer, gold_answer, estimated):
    # Exact layout observed in Astra August P5 after a second run was added.
    record = {
        "problem": "Compute the answer.",
        "gold_answer": gold_answer,
        "messages": [[{"role": "assistant", "type": "response", "content": "0"}]] * 2,
        "answers": ["0", "0"],
        "correct": [0, "TODO Grading"],
        "warnings": [0, 0],
        "detailed_costs": [
            {"cost": 0, "time": None, "input_tokens": 0, "output_tokens": 0}
            for _ in range(2)
        ],
        "judgment": [[judgment()], None],
    }
    if estimated:
        record["detailed_costs"][0]["token_usage_recovery"] = {
            "estimated": True,
            "warning": "Reasoning recovered from API timing; one failed request excluded.",
        }
    before = copy.deepcopy(record["judgment"])
    viewer.results = {"Astra": {5: record}}

    response = viewer.app.test_client().get("/view/Astra/5")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'id="run-tab-0"' in page
    assert 'id="run-tab-1"' in page
    assert viewer.get_run_judgments(record, 0)[0]["points"] == 0
    assert viewer.get_run_judgments(record, 1) == []
    assert record["judgment"] == before
    assert ("includes estimated reasoning" in page) is estimated
    assert ("Reasoning recovered from API timing" in page) is estimated
    if gold_answer is None:
        assert "First run was judged." in page


@pytest.mark.parametrize("raw", [None, [], [None], [[]], [[], None], "invalid", {}])
def test_absent_judgments_are_safe(viewer, raw):
    assert viewer.get_run_judgments({"judgment": raw}, 0) == []
    assert viewer.get_run_judgments({"judgment": raw}, 1) == []
    assert viewer.get_run_judgments({}, 0) == []


def test_legacy_judgments_and_out_of_range_runs(viewer):
    record = {"judgment": [judgment(1), None]}
    assert viewer.get_run_judgments(record, 0)[0]["judge_idx"] == 1
    assert viewer.get_run_judgments(record, 0)[0]["points"] == 1
    assert viewer.get_run_judgments(record, 1) == []
    assert viewer.get_run_judgments(record, 2) == []
    assert viewer.get_run_judgments(record, -1) == []


def test_uneven_judgment_rows_preserve_judge_and_run_indices(viewer):
    record = {"judgment": [[judgment(0)], None, [None, judgment(1)], []]}
    before = copy.deepcopy(record)
    first = viewer.get_run_judgments(record, 0)
    second = viewer.get_run_judgments(record, 1)
    assert [(x["judge_idx"], x["points"]) for x in first] == [(1, 0)]
    assert [(x["judge_idx"], x["points"]) for x in second] == [(3, 1)]
    assert record == before


@pytest.mark.parametrize("estimated", [False, True])
def test_model_totals_label_estimated_costs(viewer, monkeypatch, estimated):
    # Mixed measured/estimated runs must not present the model total as measured.
    monkeypatch.setattr(viewer, "get_problem_stats", lambda *args: {
        "nb_instances": 2, "corrects": [1, 0], "accuracy": 0.5,
        "warnings": [0, 0], "llm_annotations": [None, None],
    })
    records = {"Kimi": {1: {"cost": {"cost": 2}, "detailed_costs": [
        {"cost": 1}, {"cost": 1, "token_usage_recovery": {"estimated": estimated}},
    ]}}}
    stats = viewer.get_model_stats(records, "Kimi")
    assert stats["total_cost"] == 2
    assert viewer.model_stats_to_html(stats)["cost_estimated"] is estimated


@pytest.mark.parametrize("edit_judgment", [False, True])
@pytest.mark.parametrize("model_in_other_comp", [False, True])
def test_override_keeps_competition_after_reload(viewer, monkeypatch, edit_judgment, model_in_other_comp):
    model = "DeepSeek-V4.1-Flash (Max)"
    config = "deepseek/deepseek_v41_flash"
    record = {
        "problem": "May problem", "gold_answer": None if edit_judgment else "1",
        "messages": [[{"role": "assistant", "type": "response", "content": "1"}]],
        "answers": ["1"], "correct": [False], "llm_annotation": [True],
        "manual_overwrite": [False], "pass_at_1": 0,
        "cost": {"cost": 0},
        "detailed_costs": [{"cost": 0, "input_tokens": 0, "output_tokens": 0}],
        "judgment": [[judgment(0)]],
    }
    paths = {}
    for comp in ("test/may", "test/partial"):
        path = Path(viewer.args.output_folder) / comp / config / "1.json.zst"
        path.parent.mkdir(parents=True)
        dump_json_zst(record, path)
        paths[comp] = path
        comp_path = Path(viewer.args.competition_config_folder) / f"{comp}.yaml"
        comp_path.write_text(yaml.safe_dump({"dataset_path": "unused"}))
    viewer.all_comps = ["test/may", "test/partial"]
    viewer.args.disable_overwrite = False
    monkeypatch.setattr(viewer, "extract_existing_configs", lambda comp, *a, **kw:
        ({config: {}}, {config: model}) if comp == "test/may" or model_in_other_comp else ({}, {}))
    client = viewer.app.test_client()
    page = client.get(f"/view/{model}/1?comp=test/may")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    pattern = r'data-endpoint="([^"]*override_judgment[^"]*)"' if edit_judgment else r'action="([^"]*override/[^"]*)"'
    endpoint = unescape(re.search(pattern, html).group(1))
    assert parse_qs(urlsplit(endpoint).query)["comp"] == ["test/may"]
    other_before = paths["test/partial"].read_bytes()

    # A server reload or another tab resets the shared cache to another bench.
    viewer.select_competition("test/partial")
    form = {"points": "1", "desc": "Manually verified."} if edit_judgment else {"manual_correct": "correct"}
    response = client.post(endpoint, data=form, follow_redirects=True)
    assert response.status_code == 200
    assert response.request.args["comp"] == "test/may"
    saved = load_json_zst(paths["test/may"])
    assert saved["correct"] == [1]
    assert saved["manual_overwrite"] == [True]
    assert saved["pass_at_1"] == 1
    assert saved["messages"] == record["messages"]
    assert paths["test/partial"].read_bytes() == other_before
    if edit_judgment:
        assert saved["judgment"][0][0]["original_judgment"]["points"] == 0
    else:
        assert saved["llm_annotation"] == [None]

    # Conversation AJAX also selects its own competition after another tab changes it.
    viewer.select_competition("test/partial")
    conversation = client.get(f"/modelinteraction/{model}>>1>>0?comp=test/may")
    assert conversation.status_code == 200
    assert "Assistant" in conversation.get_data(as_text=True)


def test_stale_model_link_returns_not_found_instead_of_key_error(viewer):
    client = viewer.app.test_client()
    assert client.get("/view/missing/1").status_code == 404
    # Error teardown must release the competition lock too.
    assert client.get("/").status_code == 200


def test_refresh_redirect_preserves_competition(viewer):
    response = viewer.app.test_client().get("/refresh/test---partial/Model>>>1")
    assert response.status_code == 302
    assert parse_qs(urlsplit(response.location).query)["comp"] == ["test/partial"]
    assert urlsplit(response.location).path == "/view/Model/1"


def test_overview_and_refresh_do_not_load_unchanged_traces(viewer, monkeypatch):
    import matharena.viewer_cache as cache_module

    config = "provider/model"
    model = "Test Model"
    record = {
        "problem": "Test problem", "gold_answer": "1", "answers": ["1"],
        "correct": [True], "cost": {"cost": 0.123456},
        "messages": [[{"role": "assistant", "type": "response", "content": "Original trace"}]],
        "history": [[{"step": "solve", "timestep": 0, "messages": [
            {"role": "assistant", "type": "response", "content": "History trace"}
        ]}]],
        "detailed_costs": [{"cost": 0.123456, "input_tokens": 0, "output_tokens": 0}],
    }
    directory = Path(viewer.args.output_folder) / viewer.current_comp / config
    directory.mkdir(parents=True)
    path = directory / "1.json.zst"
    dump_json_zst(record, path)
    monkeypatch.setattr(viewer, "extract_existing_configs", lambda *a, **kw: ({config: {}}, {config: model}))
    loads = []
    original_load = cache_module.load_json_zst

    def tracked_load(path):
        loads.append(Path(path).name)
        return original_load(path)

    monkeypatch.setattr(cache_module, "load_json_zst", tracked_load)
    client = viewer.app.test_client()
    refresh_url = "/refresh/test---partial"
    assert client.get(refresh_url, follow_redirects=True).status_code == 200
    assert loads == ["1.json.zst"]
    assert "messages" not in viewer.results[model][1]
    assert "history" not in viewer.results[model][1]
    assert viewer.get_model_stats(viewer.results, model) == viewer.get_model_stats({model: {1: record}}, model)
    assert client.get(refresh_url, follow_redirects=True).status_code == 200
    assert client.get(f"/view/{model}").status_code == 200
    assert loads == ["1.json.zst"]

    assert client.get(f"/view/{model}/1").status_code == 200
    assert loads == ["1.json.zst", "1.json.zst"]
    assert viewer.get_problem_result(model, 1)["detailed_costs"][0]["cost"] == 0.123456
    assert "Original trace" in client.get(f"/modelinteraction/{model}>>1>>0").get_data(as_text=True)
    assert "History trace" in client.get(f"/historystep/{model}>>1>>0>>0>>solve").get_data(as_text=True)
    assert len(loads) == 2

    record["correct"] = [False]
    record["messages"][0][0]["content"] = "Updated trace"
    dump_json_zst(record, path)
    dump_json_zst(record, directory / "2.json.zst")
    assert client.get(refresh_url, follow_redirects=True).status_code == 200
    assert set(viewer.results[model]) == {1, 2}
    assert viewer.results[model][1]["correct"] == [False]
    assert len(loads) == 4
    assert "Updated trace" in client.get(f"/modelinteraction/{model}>>1>>0").get_data(as_text=True)
    assert len(loads) == 5

    path.unlink()
    assert client.get(refresh_url, follow_redirects=True).status_code == 200
    assert set(viewer.results[model]) == {2}
    assert client.get(f"/view/{model}/1").status_code == 404
    assert len(loads) == 5


def test_leaderboard_scores_pending_runs_and_costs_clearly(viewer):
    from bs4 import BeautifulSoup

    def record(scores, cost):
        return {"correct": scores, "cost": {"cost": cost}, "detailed_costs": []}

    viewer.results = {
        "Awaiting": {1: record(["TODO Grading"], 1)},
        "Zero": {1: record([False], 0)},
        "Beta": {1: record([0.5, 0.5], 10)},
        "Alpha": {1: record([True, "TODO Grading"], 2), 2: record([False], 4)},
        "Empty": {},
    }
    page = BeautifulSoup(viewer.app.test_client().get("/").data, "html.parser")
    rows = page.select("#leaderboard tbody tr")
    assert [row["data-model"] for row in rows] == ["Alpha", "Beta", "Zero", "Awaiting", "Empty"]
    assert [row.select_one(".rank").get_text(strip=True) for row in rows] == ["1", "1", "3", "—", "—"]
    assert rows[0]["data-cost"] == "6"
    assert rows[0]["data-unit-cost"] == "2.0"
    assert rows[0].select_one(".accuracy-number").get_text() == "50.00%"
    assert rows[0].select_one(".pending-label").get_text() == "1 pending"
    assert rows[3]["data-accuracy"] == ""
    assert rows[4]["data-unit-cost"] == ""
    assert "Total cost" in page.get_text() and "Cost / run" in page.get_text()
    assert parse_qs(urlsplit(rows[0].select_one("a")["href"]).query)["comp"] == ["test/partial"]

    model_page = BeautifulSoup(viewer.app.test_client().get("/view/Alpha").data, "html.parser")
    problems = model_page.select("#problem-results tbody tr")
    assert len(problems) == 2
    assert [row.select_one(".secondary-cost").get_text(strip=True) for row in problems] == ["$2.00", "$4.00"]
    assert "/view/Alpha/1?comp=test/partial" == problems[0].select_one("a")["href"]


def test_accuracy_keeps_equal_problem_weight_and_excludes_ungraded_values(viewer):
    viewer.results = {"Mixed": {
        1: {"correct": [True, False, "TODO Grading"], "cost": {"cost": 3}},
        2: {"correct": [True], "cost": {"cost": 1}},
        3: {"correct": [None, float("nan")], "cost": {"cost": 2}},
    }}
    stats = viewer.get_model_stats(viewer.results, "Mixed")
    assert stats["avg_accuracy"] == 0.75
    assert stats["n_graded"] == 3
    assert stats["n_pending"] == 3
    assert stats["n_solutions"] == 6
    assert stats["cost_per_run"] == 1
    assert viewer.model_stats_to_html(stats)["problem_rows"][2]["ticks"] == "❔❔"


def test_empty_leaderboard_has_an_explicit_empty_state(viewer):
    response = viewer.app.test_client().get("/")
    assert response.status_code == 200
    assert "No results yet" in response.get_data(as_text=True)


def test_token_averages_use_all_runs_without_double_counting_cached_input(viewer):
    viewer.results = {"Tokens": {
        1: {"correct": [True, False], "cost": {"cost": 3, "input_tokens": 300, "output_tokens": 50},
            "detailed_costs": [
                {"input_tokens": 100, "cached_input_tokens": 25, "output_tokens": 20},
                {"input_tokens": 200, "cached_input_tokens": 100, "output_tokens": 30},
            ]},
        2: {"correct": [True], "cost": {"cost": 1}, "detailed_costs": [
            {"input_tokens": 100, "cached_input_tokens": 90, "output_tokens": 15},
        ]},
    }}
    stats = viewer.get_model_stats(viewer.results, "Tokens")
    assert stats["tokens_per_run"] == {
        "input_tokens": 400 / 3, "cached_input_tokens": 215 / 3, "output_tokens": 65 / 3,
    }
    row = viewer.model_stats_to_html(stats)["problem_rows"][0]
    assert row["tokens_per_run"]["input_tokens"] == 150
    assert row["tokens_per_run"]["cached_input_tokens"] == 62.5
    assert row["cost_per_run"] == "1.50"
    assert "Cached input / run" in viewer.app.test_client().get("/").get_data(as_text=True)
    assert viewer.get_token_totals({"cost": {"input_tokens": 0}}, 1) == {
        "input_tokens": 0, "cached_input_tokens": None, "output_tokens": None,
    }


def test_consecutive_cot_messages_share_a_dropdown_without_reordering(viewer):
    from bs4 import BeautifulSoup

    conversation = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "type": "cot", "content": "Reasoning one"},
        {"role": "assistant", "type": "cot", "content": "Reasoning two <script>alert(1)</script>"},
        {"role": "assistant", "type": "tool_call", "tool_name": "python", "tool_call_id": "call-1", "arguments": {"code": "print(1)"}},
        {"role": "tool_response", "tool_name": "python", "tool_call_id": "call-1", "content": "Tool result"},
        {"role": "assistant", "type": "cot", "content": "Reasoning three"},
        {"role": "assistant", "type": "cot", "content": [{"type": "text", "text": "Reasoning four"}, {"type": "image_url", "image_url": {"url": "example.png"}}]},
        {"role": "assistant", "type": "response", "content": "Full final answer"},
    ]
    before = copy.deepcopy(conversation)
    with viewer.app.test_request_context():
        html = viewer.render_conversation_html(conversation)
    page = BeautifulSoup(html, "html.parser")
    groups = page.select(".cot-message")
    assert len(groups) == 2
    assert all("open" not in group.attrs for group in groups)
    assert [len(group.select(".cot-part")) for group in groups] == [2, 2]
    assert len(groups[0].select(".box")) == 1
    assert groups[1].select_one("img")["src"] == "example.png"
    assert not page.select("script")
    text = page.get_text()
    assert text.index("Reasoning two") < text.index("print(1)") < text.index("Tool result") < text.index("Reasoning three") < text.index("Full final answer")
    assert conversation == before


@pytest.mark.parametrize("gold_answer", ["42", None])
def test_final_answer_and_judgment_are_outside_the_collapsed_trace(viewer, gold_answer):
    from bs4 import BeautifulSoup

    conversation = [
        {"role": "user", "content": "Original prompt"},
        {"role": "assistant", "type": "cot", "content": "Private trace reasoning"},
        {"role": "assistant", "type": "response", "content": "Intermediate response"},
        {"role": "assistant", "type": "response", "content": "Complete final answer, with its explanation."},
    ]
    viewer.results = {"Model": {1: {
        "problem": "Question", "gold_answer": gold_answer, "messages": [conversation],
        "answers": ["42"], "correct": [True], "cost": {"cost": 1},
        "detailed_costs": [{"cost": 1, "input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 10}],
        "judgment": [[judgment(1)]],
    }}}
    client = viewer.app.test_client()
    page = BeautifulSoup(client.get("/view/Model/1").data, "html.parser")
    trace = page.select_one(".trace-details")
    assert "open" not in trace.attrs
    assert page.select_one(".final-answer").get_text(strip=True) == "Final answerComplete final answer, with its explanation."
    assert page.select_one(".final-answer").find_parent("details") is None
    assert page.select_one(".judgment-details").has_attr("open")
    assert page.select_one(".judgment-details").find_parent("details") is None
    assert "Private trace reasoning" not in page.get_text()
    assert "Cached input80" in page.select_one(".run-metrics").get_text(strip=True)
    assert not page.select(".brand, .sidebar-footer, .topbar, .eyebrow, .page-heading, .page-description")
    response = client.get("/modelinteraction/Model>>1>>0?trace_only=1").get_data(as_text=True)
    assert "Private trace reasoning" in response
    assert "Intermediate response" in response
    assert "Complete final answer" not in response
    assert conversation[-1]["content"] == "Complete final answer, with its explanation."
