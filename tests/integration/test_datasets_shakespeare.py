from __future__ import annotations

# New strict API imports
from ml_playground.configuration.models import PreparerConfig
from ml_playground.experiments.shakespeare.preparer import ShakespearePreparer


class _StubTokenizer:
    """Lightweight tokenizer satisfying the Tokenizer protocol for tests."""

    def __init__(self) -> None:
        self._calls: list[str] = []
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    @property
    def name(self) -> str:
        return "stub"

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        self._calls.append(text)
        # Build deterministic stoi/itos on the fly to maintain reversible mapping.
        for ch in text:
            if ch not in self.stoi:
                idx = len(self.stoi)
                self.stoi[ch] = idx
                self.itos[idx] = ch
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids):  # type: ignore[override]
        return "".join(self.itos.get(int(i), "") for i in token_ids)

    def pop_calls(self) -> list[str]:
        calls = list(self._calls)
        self._calls.clear()
        return calls


def test_shakespeare_download_and_encode(tmp_path) -> None:
    """Test shakespeare preparer downloads, splits, encodes, and writes outputs without mocks."""
    test_text = "Hello world! This is test data for Shakespeare."

    # Arrange: set base_dir via DI
    exp_dir = tmp_path / "experiments" / "shakespeare"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = exp_dir / "datasets"

    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:  # no-op
            return None

    http_calls: list[str] = []

    def _http_get(url: str, timeout: int = 30):
        http_calls.append(url)
        return _Resp(test_text)

    tokenizer = _StubTokenizer()

    writer_called: dict[str, object] = {"called": False, "args": None}

    def _writer(path, train_ids, val_ids, meta, logger=None):
        writer_called["called"] = True
        writer_called["args"] = (path, train_ids, val_ids, meta)

    cfg = PreparerConfig()
    cfg.extras.update(
        {
            "base_dir": exp_dir,
            "http_get": _http_get,
            "tokenizer_factory": lambda: tokenizer,
            "writer_fn": _writer,
        }
    )

    report = ShakespearePreparer().prepare(cfg)

    # Assert: download occurred and input file written
    assert http_calls, "http_get should be called"
    assert (ds_dir / "input.txt").exists()
    # Tokenizer used twice (train/val)
    assert len(tokenizer.pop_calls()) == 2
    # Writer called and received ds_dir
    assert writer_called["called"] is True
    args_obj = writer_called["args"]
    assert isinstance(args_obj, tuple)
    assert args_obj[0] == ds_dir
    # Report includes created or updated files tuples
    assert hasattr(report, "created_files") and hasattr(report, "messages")


def test_shakespeare_skip_download_if_exists(tmp_path) -> None:
    """Test preparer skips download when input file exists without network call."""
    exp_dir = tmp_path / "experiments" / "shakespeare"
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "input.txt").write_text("Existing Shakespeare data.")

    def _http_get(_url: str, timeout: int = 30):
        raise AssertionError("http_get should not be called when input exists")

    tok = _StubTokenizer()
    writer_called = {"n": 0}

    def _writer(*a, **k):
        writer_called["n"] += 1

    cfg = PreparerConfig()
    cfg.extras.update(
        {
            "base_dir": exp_dir,
            "http_get": _http_get,
            "tokenizer_factory": lambda: tok,
            "writer_fn": _writer,
        }
    )

    ShakespearePreparer().prepare(cfg)

    assert writer_called["n"] == 1
    assert len(tok.pop_calls()) == 2


def test_shakespeare_data_split_ratios(tmp_path) -> None:
    """Test that data is split into 90% train, 10% val before encoding without mocks."""
    exp_dir = tmp_path / "experiments" / "shakespeare"
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    test_text = "x" * 100
    (ds_dir / "input.txt").write_text(test_text)

    tok = _StubTokenizer()

    cfg = PreparerConfig()
    cfg.extras.update(
        {
            "base_dir": exp_dir,
            "tokenizer_factory": lambda: tok,
            "writer_fn": lambda *a, **k: None,
        }
    )

    ShakespearePreparer().prepare(cfg)

    calls = tok.pop_calls()
    assert len(calls) == 2
    train_text, val_text = calls
    assert len(train_text) == 90
    assert len(val_text) == 10
