from __future__ import annotations

import contextlib
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
import requests.exceptions

import ml_playground.experiments.shakespeare.preparer as shakespeare_module
from ml_playground.configuration.models import PreparerConfig
from ml_playground.core.error_handling import DataError
from ml_playground.experiments.shakespeare.preparer import ShakespearePreparer


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.vocab_size = 256
        self.name = "fake"

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        return list(range(len(text)))

    def decode(self, ids: list[int]) -> str:
        return "".join(chr((i % 26) + 97) for i in ids)


class _FakeResponse:
    def __init__(self, *, text: str | None, ok: bool = True) -> None:
        self.text = text
        self._ok = ok
        self.called = False

    def raise_for_status(self) -> None:
        if not self._ok:
            raise requests.exceptions.HTTPError("boom")
        self.called = True


def _make_cfg(
    base_dir: Path,
    *,
    http_response: _FakeResponse | None = None,
    http_error: Exception | None = None,
    writer_calls: list[dict[str, Any]] | None = None,
    tokenizer: _FakeTokenizer | None = None,
) -> PreparerConfig:
    def _http_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        if http_error:
            raise http_error
        assert http_response is not None
        return http_response

    tok = tokenizer or _FakeTokenizer()

    def _writer(
        ds_dir: Path,
        train_ids,
        val_ids,
        meta,
        *,
        logger,
    ) -> None:
        if writer_calls is not None:
            writer_calls.append(
                {
                    "ds_dir": ds_dir,
                    "train_len": len(train_ids),
                    "val_len": len(val_ids),
                    "meta": meta,
                    "logger": logger,
                }
            )
        (ds_dir / "train.bin").write_bytes(bytes(train_ids))
        (ds_dir / "val.bin").write_bytes(bytes(val_ids))
        (ds_dir / "meta.pkl").write_text(str(meta), encoding="utf-8")

    extras: dict[str, Any] = {
        "base_dir": str(base_dir),
        "http_get": _http_get if http_response or http_error else None,
        "tokenizer_factory": lambda: tok,
    }
    if writer_calls is not None:
        extras["writer_fn"] = _writer

    # Remove None extras to avoid confusing preparer logic
    extras = {k: v for k, v in extras.items() if v is not None}

    return PreparerConfig(
        tokenizer_type="tiktoken",
        logger=logging.getLogger("shakespeare-test"),
        extras=extras,
    )


def test_shakespeare_preparer_downloads_when_missing(tmp_path: Path) -> None:
    base_dir = tmp_path / "shakespeare"
    base_dir.mkdir()
    ds_dir = base_dir / "datasets"

    response = _FakeResponse(text="romeo juliet")
    writer_calls: list[dict[str, Any]] = []
    tokenizer = _FakeTokenizer()
    cfg = _make_cfg(
        base_dir,
        http_response=response,
        writer_calls=writer_calls,
        tokenizer=tokenizer,
    )

    report = ShakespearePreparer().prepare(cfg)

    assert (ds_dir / "train.bin").exists()
    assert (ds_dir / "val.bin").exists()
    assert (ds_dir / "meta.pkl").exists()
    assert response.called is True
    assert writer_calls
    assert writer_calls[0]["train_len"] > 0
    assert writer_calls[0]["val_len"] > 0
    assert len(tokenizer.calls) == 2
    assert any("prepared dataset" in msg for msg in report.messages)


def test_shakespeare_preparer_uses_module_directory(tmp_path: Path) -> None:
    exp_dir = tmp_path / "shakespeare_default"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()
    input_file = ds_dir / "input.txt"
    input_file.write_text("to be or not to be", encoding="utf-8")

    tokenizer = _FakeTokenizer()

    cfg = PreparerConfig(
        tokenizer_type="tiktoken",
        logger=logging.getLogger("shakespeare-default"),
        extras={},
    )

    original_file = shakespeare_module.__file__
    original_tokenizer_factory = shakespeare_module.create_tokenizer
    shakespeare_module.__file__ = str(exp_dir / "preparer.py")
    shakespeare_module.create_tokenizer = lambda *_a, **_k: tokenizer

    try:
        report = ShakespearePreparer().prepare(cfg)
        assert (ds_dir / "train.bin").exists()
        assert (ds_dir / "val.bin").exists()
        assert (ds_dir / "meta.pkl").exists()
        assert len(tokenizer.calls) == 2
        assert any("prepared dataset" in msg for msg in report.messages)
    finally:
        shakespeare_module.__file__ = original_file
        shakespeare_module.create_tokenizer = original_tokenizer_factory
        with contextlib.suppress(FileNotFoundError):
            for path in sorted(ds_dir.glob("*")):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            ds_dir.rmdir()
        with contextlib.suppress(FileNotFoundError):
            input_file.unlink()


def test_shakespeare_preparer_http_failure_raises(tmp_path: Path) -> None:
    base_dir = tmp_path / "http_failure"
    base_dir.mkdir()
    cfg = _make_cfg(
        base_dir,
        http_error=requests.exceptions.Timeout("timeout"),
    )
    with pytest.raises(DataError) as excinfo:
        ShakespearePreparer().prepare(cfg)
    assert "Failed to download Shakespeare dataset" in str(excinfo.value)


def test_shakespeare_preparer_missing_text_attribute(tmp_path: Path) -> None:
    base_dir = tmp_path / "missing_text"
    base_dir.mkdir()
    response = _FakeResponse(text=None)
    cfg = _make_cfg(base_dir, http_response=response)
    with pytest.raises(DataError) as excinfo:
        ShakespearePreparer().prepare(cfg)
    assert "http_get did not return an object with .text" in str(excinfo.value)


def test_shakespeare_preparer_non_callable_hooks(tmp_path: Path) -> None:
    base_dir = tmp_path / "fallback"
    base_dir.mkdir()
    ds_dir = base_dir / "datasets"

    class MinimalResponse:
        def __init__(self, text: str) -> None:
            self.text = text

    extras = {
        "base_dir": str(base_dir),
        "http_get": lambda *_args, **_kwargs: MinimalResponse("all the world's a stage"),
        "tokenizer_factory": "noop",
        "writer_fn": "noop",
    }

    cfg = PreparerConfig(
        tokenizer_type="tiktoken",
        logger=logging.getLogger("shakespeare-fallback"),
        extras=extras,
    )

    tokenizer = _FakeTokenizer()
    original_tokenizer_factory = shakespeare_module.create_tokenizer
    shakespeare_module.create_tokenizer = lambda *_a, **_k: tokenizer

    try:
        report = ShakespearePreparer().prepare(cfg)
    finally:
        shakespeare_module.create_tokenizer = original_tokenizer_factory

    assert (ds_dir / "train.bin").exists()
    assert (ds_dir / "val.bin").exists()
    assert (ds_dir / "meta.pkl").exists()
    assert len(tokenizer.calls) == 2
    assert any("prepared dataset" in msg for msg in report.messages)


def test_shakespeare_preparer_handles_config_without_extras(tmp_path: Path) -> None:
    exp_dir = tmp_path / "shakespeare_stub"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()
    input_file = ds_dir / "input.txt"
    input_file.write_text("friends romans countrymen", encoding="utf-8")

    class _StubCfg:
        def __init__(self) -> None:
            self.tokenizer_type = "tiktoken"
            self.logger = logging.getLogger("shakespeare-stub")
            self.extras = None
            self.raw_text_path = None

    cfg = _StubCfg()
    tokenizer = _FakeTokenizer()

    original_file = shakespeare_module.__file__
    original_tokenizer_factory = shakespeare_module.create_tokenizer
    shakespeare_module.__file__ = str(exp_dir / "preparer.py")
    shakespeare_module.create_tokenizer = lambda *_a, **_k: tokenizer

    try:
        report = ShakespearePreparer().prepare(cfg)  # type: ignore[arg-type]
        assert (ds_dir / "train.bin").exists()
        assert (ds_dir / "val.bin").exists()
        assert (ds_dir / "meta.pkl").exists()
        assert len(tokenizer.calls) == 2
        assert any("prepared dataset" in msg for msg in report.messages)
    finally:
        shakespeare_module.__file__ = original_file
        shakespeare_module.create_tokenizer = original_tokenizer_factory
        with contextlib.suppress(FileNotFoundError):
            for path in sorted(ds_dir.glob("*")):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            ds_dir.rmdir()
        with contextlib.suppress(FileNotFoundError):
            input_file.unlink()


def test_shakespeare_preparer_default_http_get(tmp_path: Path) -> None:
    exp_dir = tmp_path / "shakespeare_http"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"

    class ResponseWithRaise:
        def __init__(self, text: str) -> None:
            self.text = text
            self._called = False

        def raise_for_status(self) -> None:
            self._called = True

    tokenizer = _FakeTokenizer()
    response = ResponseWithRaise("lend me your ears")

    cfg = PreparerConfig(
        tokenizer_type="tiktoken",
        logger=logging.getLogger("shakespeare-http"),
        extras={},
    )

    original_file = shakespeare_module.__file__
    original_tokenizer_factory = shakespeare_module.create_tokenizer
    original_requests_get = shakespeare_module.requests.get
    shakespeare_module.__file__ = str(exp_dir / "preparer.py")
    shakespeare_module.create_tokenizer = lambda *_a, **_k: tokenizer
    shakespeare_module.requests.get = lambda *_a, **_k: response

    try:
        report = ShakespearePreparer().prepare(cfg)
        assert (ds_dir / "train.bin").exists()
        assert (ds_dir / "val.bin").exists()
        assert (ds_dir / "meta.pkl").exists()
        assert len(tokenizer.calls) == 2
        assert response._called is True
        assert any("prepared dataset" in msg for msg in report.messages)
    finally:
        shakespeare_module.__file__ = original_file
        shakespeare_module.create_tokenizer = original_tokenizer_factory
        shakespeare_module.requests.get = original_requests_get
        with contextlib.suppress(FileNotFoundError):
            for path in sorted(ds_dir.glob("*")):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            ds_dir.rmdir()
