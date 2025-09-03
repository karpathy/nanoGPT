from __future__ import annotations

from types import SimpleNamespace

import pytest

import ml_playground.experiments as exps


def test_load_preparers_returns_if_already_populated(monkeypatch: pytest.MonkeyPatch):
    # Pre-populate
    exps.PREPARERS.clear()
    exps.PREPARERS["foo"] = lambda: None

    # If resources.files is called, fail the test
    def bad_files(_):  # noqa: D401
        raise AssertionError(
            "resources.files should not be called when PREPARERS present"
        )

    monkeypatch.setattr(exps, "resources", SimpleNamespace(files=bad_files))
    exps.load_preparers()
    assert "foo" in exps.PREPARERS


def test_load_preparers_handles_resources_error(monkeypatch: pytest.MonkeyPatch):
    exps.PREPARERS.clear()

    def raise_files(_):  # noqa: D401
        raise RuntimeError("boom")

    monkeypatch.setattr(exps, "resources", SimpleNamespace(files=raise_files))
    exps.load_preparers()
    assert exps.PREPARERS == {}


class _FakePath:
    def __init__(self, is_file: bool):
        self._is_file = is_file

    def is_file(self) -> bool:  # noqa: D401
        return self._is_file


class _FakeEntry:
    def __init__(self, name: str, is_dir: bool, has_preparer: bool):
        self.name = name
        self._is_dir = is_dir
        self._has_preparer = has_preparer

    def is_dir(self) -> bool:  # noqa: D401
        return self._is_dir

    def __truediv__(self, other: str):  # noqa: D401
        if other == "preparer.py":
            return _FakePath(self._has_preparer)
        raise AssertionError("unexpected path component")


class _FakeRoot:
    def __init__(self, entries):
        self._entries = entries

    def iterdir(self):  # noqa: D401
        for e in self._entries:
            yield e


def test_load_preparers_registers_class(monkeypatch: pytest.MonkeyPatch):
    exps.PREPARERS.clear()
    root = _FakeRoot([_FakeEntry("expA", True, True)])
    monkeypatch.setattr(exps, "resources", SimpleNamespace(files=lambda _: root))

    class Prep:
        def prepare(self, *_args, **_kwargs):  # noqa: D401
            # no-op
            return None

    # import_module should return module with class Prep
    fake_mod = SimpleNamespace(Prep=Prep)
    monkeypatch.setattr(exps, "import_module", lambda name: fake_mod)

    exps.load_preparers()
    assert "expA" in exps.PREPARERS
    # Calling the registered function shouldn't raise
    exps.PREPARERS["expA"]()


def test_load_preparers_raises_on_import_failure(monkeypatch: pytest.MonkeyPatch):
    exps.PREPARERS.clear()
    root = _FakeRoot([_FakeEntry("bad", True, True)])
    monkeypatch.setattr(exps, "resources", SimpleNamespace(files=lambda _: root))
    monkeypatch.setattr(
        exps, "import_module", lambda name: (_ for _ in ()).throw(RuntimeError("nope"))
    )

    with pytest.raises(SystemExit) as ei:
        exps.load_preparers()
    assert "Failed to load experiment 'bad':" in str(ei.value)


def test_load_preparers_skips_non_dir_and_missing_preparer(
    monkeypatch: pytest.MonkeyPatch,
):
    exps.PREPARERS.clear()
    # One non-dir, one dir without preparer.py
    root = _FakeRoot(
        [
            _FakeEntry("file.txt", False, False),
            _FakeEntry("expNoPrep", True, False),
        ]
    )
    monkeypatch.setattr(exps, "resources", SimpleNamespace(files=lambda _: root))
    exps.load_preparers()
    assert exps.PREPARERS == {}


def test_load_preparers_module_without_prepare_class(monkeypatch: pytest.MonkeyPatch):
    exps.PREPARERS.clear()
    root = _FakeRoot([_FakeEntry("expNoClass", True, True)])
    monkeypatch.setattr(exps, "resources", SimpleNamespace(files=lambda _: root))

    # Module has classes but none with 'prepare'
    class X:  # noqa: D401
        pass

    fake_mod = SimpleNamespace(X=X)
    monkeypatch.setattr(exps, "import_module", lambda name: fake_mod)
    exps.load_preparers()
    # Should not register anything
    assert exps.PREPARERS == {}


def test_load_preparers_noarg_prepare_calls_without_args(
    monkeypatch: pytest.MonkeyPatch,
):
    exps.PREPARERS.clear()
    root = _FakeRoot([_FakeEntry("expNoArg", True, True)])
    monkeypatch.setattr(exps, "resources", SimpleNamespace(files=lambda _: root))

    prepared = {"called": False}

    class Prep:
        def prepare(self):  # noqa: D401
            prepared["called"] = True

    fake_mod = SimpleNamespace(Prep=Prep)
    monkeypatch.setattr(exps, "import_module", lambda name: fake_mod)
    exps.load_preparers()
    assert "expNoArg" in exps.PREPARERS
    exps.PREPARERS["expNoArg"]()
    assert prepared["called"] is True


def test_load_preparers_catches_per_entry_exception(monkeypatch: pytest.MonkeyPatch):
    exps.PREPARERS.clear()

    class _BoomEntry:
        def __init__(self):
            self.name = "boom"

        def is_dir(self):  # noqa: D401
            raise RuntimeError("boom")

    root = _FakeRoot([_BoomEntry(), _FakeEntry("ok", True, False)])
    monkeypatch.setattr(exps, "resources", SimpleNamespace(files=lambda _: root))
    # Should not raise
    exps.load_preparers()
    # No registrations since second entry had no preparer
    assert exps.PREPARERS == {}
