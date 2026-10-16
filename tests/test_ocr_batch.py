import pytest

from matharena.tools import paper_search


def test_ocr_batch_flattens_pages_across_pdfs_and_writes_outputs(tmp_path, monkeypatch):
    page_counts = {
        str(tmp_path / "paper-a.pdf"): 2,
        str(tmp_path / "paper-b.pdf"): 3,
    }
    closed_docs = []

    class FakeDoc:
        def __init__(self, path):
            self.path = path
            self.page_count = page_counts[path]

        def close(self):
            closed_docs.append(self.path)

    fake_clients = []

    class FakeAPIClient:
        def __init__(self, **_kwargs):
            self.concurrent_requests = 2
            self.chunk_sizes = []
            fake_clients.append(self)

        def run_queries(self, queries, no_tqdm=False):
            self.chunk_sizes.append(len(queries))
            for idx, query in enumerate(queries):
                image_url = query[0]["content"][0]["image_url"]["url"]
                yield idx, [{"role": "assistant", "content": f"markdown for {image_url}"}], {}

    config_path = tmp_path / "ocr.yaml"
    config_path.write_text("model: fake\nhuman_readable_id: Fake\ndate: '2025-01-01'\n", encoding="utf-8")
    monkeypatch.setattr(paper_search, "MODEL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(paper_search, "APIClient", FakeAPIClient)
    monkeypatch.setattr(paper_search.fitz, "open", lambda path: FakeDoc(path))
    monkeypatch.setattr(paper_search, "pdf_page_to_pil", lambda doc, page_idx, dpi=300: f"{doc.path}:{page_idx}:{dpi}")
    monkeypatch.setattr(paper_search, "pil_to_data_uri", lambda img: f"data:{img}")

    output_paths = paper_search.ocr_batch(
        [
            (str(tmp_path / "paper-a.pdf"), "paper-a"),
            (str(tmp_path / "paper-b.pdf"), "paper-b"),
        ],
        store_folder=str(tmp_path / "ocr"),
        page_batch_size=3,
        dpi=200,
    )

    assert fake_clients[0].chunk_sizes == [3, 2]
    assert set(output_paths) == {"paper-a", "paper-b"}
    assert closed_docs == [str(tmp_path / "paper-a.pdf"), str(tmp_path / "paper-b.pdf")]

    paper_a = (tmp_path / "ocr" / "paper-a.md").read_text(encoding="utf-8")
    assert "##### Page 1 #####" in paper_a
    assert f"markdown for data:{tmp_path / 'paper-a.pdf'}:0:200" in paper_a
    assert "##### Page 2 #####" in paper_a
    assert f"markdown for data:{tmp_path / 'paper-a.pdf'}:1:200" in paper_a

    paper_b = (tmp_path / "ocr" / "paper-b.md").read_text(encoding="utf-8")
    assert "##### Page 1 #####" in paper_b
    assert f"markdown for data:{tmp_path / 'paper-b.pdf'}:0:200" in paper_b
    assert "##### Page 3 #####" in paper_b
    assert f"markdown for data:{tmp_path / 'paper-b.pdf'}:2:200" in paper_b


def test_ocr_batch_writes_completed_pdf_before_later_failure(tmp_path, monkeypatch):
    page_counts = {
        str(tmp_path / "paper-a.pdf"): 2,
        str(tmp_path / "paper-b.pdf"): 2,
    }

    class FakeDoc:
        def __init__(self, path):
            self.path = path
            self.page_count = page_counts[path]

        def close(self):
            pass

    class FakeAPIClient:
        def __init__(self, **_kwargs):
            self.concurrent_requests = 1
            self.calls = 0

        def run_queries(self, queries, no_tqdm=False):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("server died")
            for idx, query in enumerate(queries):
                image_url = query[0]["content"][0]["image_url"]["url"]
                yield idx, [{"role": "assistant", "content": f"markdown for {image_url}"}], {}

    config_path = tmp_path / "ocr.yaml"
    config_path.write_text("model: fake\n", encoding="utf-8")
    monkeypatch.setattr(paper_search, "MODEL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(paper_search, "APIClient", FakeAPIClient)
    monkeypatch.setattr(paper_search.fitz, "open", lambda path: FakeDoc(path))
    monkeypatch.setattr(paper_search, "pdf_page_to_pil", lambda doc, page_idx, dpi=300: f"{doc.path}:{page_idx}:{dpi}")
    monkeypatch.setattr(paper_search, "pil_to_data_uri", lambda img: f"data:{img}")

    with pytest.raises(RuntimeError, match="server died"):
        paper_search.ocr_batch(
            [
                (str(tmp_path / "paper-a.pdf"), "paper-a"),
                (str(tmp_path / "paper-b.pdf"), "paper-b"),
            ],
            store_folder=str(tmp_path / "ocr"),
            page_batch_size=2,
        )

    paper_a_path = tmp_path / "ocr" / "paper-a.md"
    assert paper_a_path.exists()
    paper_a = paper_a_path.read_text(encoding="utf-8")
    assert f"markdown for data:{tmp_path / 'paper-a.pdf'}:0:300" in paper_a
    assert f"markdown for data:{tmp_path / 'paper-a.pdf'}:1:300" in paper_a
    assert not (tmp_path / "ocr" / "paper-b.md").exists()
