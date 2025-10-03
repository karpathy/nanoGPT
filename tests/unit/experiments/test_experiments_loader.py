from __future__ import annotations

from types import SimpleNamespace

import pytest

import ml_playground.experiments.registry as registry


def test_load_preparers_returns_if_already_populated():
    # Pre-populate
    registry.PREPARERS.clear()
    registry.PREPARERS["foo"] = lambda: None

    # If resources.files is called, fail the test
    def bad_files(_):  # noqa: D401
        raise AssertionError(
            "resources.files should not be called when PREPARERS present"
        )

    registry.load_preparers(resources_mod=SimpleNamespace(files=bad_files))
    assert "foo" in registry.PREPARERS


def test_load_preparers_handles_resources_error():
    registry.PREPARERS.clear()

    def raise_files(_):  # noqa: D401
        raise RuntimeError("boom")

    registry.load_preparers(resources_mod=SimpleNamespace(files=raise_files))
    assert registry.PREPARERS == {}


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


def test_load_preparers_registers_class():
    registry.PREPARERS.clear()
    root = _FakeRoot([_FakeEntry("expA", True, True)])
    resources_ns = SimpleNamespace(files=lambda _: root)

    class Prep:
        def prepare(self, *_args, **_kwargs):  # noqa: D401
            # no-op
            return None

    # import_module should return module with class Prep
    fake_mod = SimpleNamespace(Prep=Prep)
    registry.load_preparers(
        resources_mod=resources_ns, import_mod=lambda name: fake_mod
    )
    assert "expA" in registry.PREPARERS
    # Calling the registered function shouldn't raise
    registry.PREPARERS["expA"]()


def test_load_preparers_raises_on_import_failure():
    registry.PREPARERS.clear()
    root = _FakeRoot([_FakeEntry("bad", True, True)])
    resources_ns = SimpleNamespace(files=lambda _: root)

    def bad_import(name: str):  # noqa: D401
        return (_ for _ in ()).throw(RuntimeError("nope"))

    with pytest.raises(SystemExit) as ei:
        registry.load_preparers(resources_mod=resources_ns, import_mod=bad_import)
    assert "Failed to load experiment 'bad':" in str(ei.value)


def test_load_preparers_skips_non_dir_and_missing_preparer():
    registry.PREPARERS.clear()
    # One non-dir, one dir without preparer.py
    root = _FakeRoot(
        [
            _FakeEntry("file.txt", False, False),
            _FakeEntry("expNoPrep", True, False),
        ]
    )
    registry.load_preparers(resources_mod=SimpleNamespace(files=lambda _: root))
    assert registry.PREPARERS == {}


def test_load_preparers_module_without_prepare_class():
    registry.PREPARERS.clear()
    root = _FakeRoot([_FakeEntry("expNoClass", True, True)])
    resources_ns = SimpleNamespace(files=lambda _: root)

    # Module has classes but none with 'prepare'
    class X:  # noqa: D401
        pass

    fake_mod = SimpleNamespace(X=X)
    registry.load_preparers(
        resources_mod=resources_ns, import_mod=lambda name: fake_mod
    )
    # Should not register anything
    assert registry.PREPARERS == {}


def test_load_preparers_noarg_prepare_calls_without_args():
    registry.PREPARERS.clear()
    root = _FakeRoot([_FakeEntry("expNoArg", True, True)])
    resources_ns = SimpleNamespace(files=lambda _: root)

    prepared = {"called": False}

    class Prep:
        def prepare(self):  # noqa: D401
            prepared["called"] = True

    fake_mod = SimpleNamespace(Prep=Prep)
    registry.load_preparers(
        resources_mod=resources_ns, import_mod=lambda name: fake_mod
    )
    assert "expNoArg" in registry.PREPARERS
    registry.PREPARERS["expNoArg"]()
    assert prepared["called"] is True


def test_load_preparers_catches_per_entry_exception():
    registry.PREPARERS.clear()

    class _BoomEntry:
        def __init__(self):
            self.name = "boom"

        def is_dir(self):  # noqa: D401
            raise RuntimeError("boom")

    root = _FakeRoot([_BoomEntry(), _FakeEntry("ok", True, False)])
    resources_ns = SimpleNamespace(files=lambda _: root)
    # Should not raise
    registry.load_preparers(resources_mod=resources_ns)
    # No registrations since second entry had no preparer
    assert registry.PREPARERS == {}
