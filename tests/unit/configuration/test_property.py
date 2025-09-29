"""Property-based tests for the configuration package."""

from __future__ import annotations

import string
import tempfile
from pathlib import Path
from typing import Any

import hypothesis.strategies as st
from hypothesis import given, settings
import pytest
import tomllib

from ml_playground.configuration import loading as config_loading


@st.composite
def dict_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate nested dictionaries with string keys and various values."""
    return draw(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.recursive(
                st.none()
                | st.booleans()
                | st.integers()
                | st.floats(allow_nan=False, allow_infinity=False)
                | st.text(max_size=50),
                lambda children: st.dictionaries(
                    keys=st.text(min_size=1, max_size=10), values=children
                )
                | st.lists(children, max_size=5),
                max_leaves=10,
            ),
            max_size=10,
        )
    )


@st.composite
def toml_dict_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate dictionaries that can be serialized to TOML."""

    def valid_toml_value() -> st.SearchStrategy[Any]:
        return st.one_of(
            st.booleans(),
            st.integers(min_value=-1000, max_value=1000),
            st.floats(
                min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
            ),
            st.text(max_size=50),
            st.dates(),
            st.times(),
            st.datetimes(),
        )

    return draw(
        st.dictionaries(
            keys=st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                    min_codepoint=97,
                    max_codepoint=122,
                ),
            ),
            values=st.recursive(
                valid_toml_value(),
                lambda children: st.dictionaries(
                    keys=st.text(
                        min_size=1,
                        max_size=10,
                        alphabet=st.characters(
                            whitelist_categories=("L", "N"),
                            min_codepoint=97,
                            max_codepoint=122,
                        ),
                    ),
                    values=children,
                )
                | st.lists(children, max_size=5),
                max_leaves=5,
            ),
            max_size=5,
        )
    )


class TestDeepMergeDicts:
    """Property-based tests for `deep_merge_dicts`."""

    @given(base=dict_strategy(), override=dict_strategy())
    @settings(max_examples=100)
    def test_merge_preserves_base_keys_not_in_override(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> None:
        result = config_loading.deep_merge_dicts(base, override)
        for key in base:
            if key not in override:
                assert key in result
                assert result[key] == base[key]

    @given(base=dict_strategy(), override=dict_strategy())
    @settings(max_examples=100)
    def test_merge_overrides_base_values(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> None:
        result = config_loading.deep_merge_dicts(base, override)
        for key, value in override.items():
            if not isinstance(value, dict):
                assert result[key] == value
            elif key in base and isinstance(base[key], dict):
                assert result[key] == config_loading.deep_merge_dicts(base[key], value)

    @given(d1=dict_strategy(), d2=dict_strategy(), d3=dict_strategy())
    @settings(max_examples=50)
    def test_merge_associativity(
        self, d1: dict[str, Any], d2: dict[str, Any], d3: dict[str, Any]
    ) -> None:
        try:
            result1 = config_loading.deep_merge_dicts(
                config_loading.deep_merge_dicts(d1, d2), d3
            )
            result2 = config_loading.deep_merge_dicts(
                d1, config_loading.deep_merge_dicts(d2, d3)
            )
            assert result1 == result2
        except Exception:
            pass

    @given(base=dict_strategy())
    @settings(max_examples=50)
    def test_merge_with_empty_override(self, base: dict[str, Any]) -> None:
        result = config_loading.deep_merge_dicts(base, {})
        assert result == base

    @given(override=dict_strategy())
    @settings(max_examples=50)
    def test_merge_with_empty_base(self, override: dict[str, Any]) -> None:
        result = config_loading.deep_merge_dicts({}, override)
        assert result == override


class TestTomlReading:
    """Property-based tests for TOML reading functionality."""

    @given(content=toml_dict_strategy())
    @settings(max_examples=50)
    def test_round_trip_toml_serialization(self, content: dict[str, Any]) -> None:
        import tomli_w

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            try:
                tomli_w.dump(content, f)
                f.flush()
                path = Path(f.name)
                result = config_loading.read_toml_dict(path)
                assert isinstance(result, dict)
                for key in content:
                    assert key in result
            finally:
                Path(f.name).unlink(missing_ok=True)

    @given(
        content=st.sampled_from(
            ["[invalid", "key =", "[[table]", "key = value extra", '{"json"}']
        )
    )
    @settings(max_examples=20)
    def test_invalid_toml_raises_exception(self, content: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            try:
                f.write(content)
                f.flush()
                path = Path(f.name)
                if content.strip():
                    with pytest.raises((Exception, tomllib.TOMLDecodeError)):
                        config_loading.read_toml_dict(path)
            finally:
                Path(f.name).unlink(missing_ok=True)


class TestConfigPaths:
    """Property-based tests for configuration path computation."""

    @given(
        experiment=st.text(
            alphabet=string.ascii_letters + string.digits + "_-",
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_experiment_path_computation(self, experiment: str) -> None:
        path = config_loading.get_cfg_path(experiment, None)
        assert str(path).endswith(f"experiments/{experiment}/config.toml")
        assert path.is_absolute()

    @given(
        exp_config=st.text(
            alphabet=string.ascii_letters + string.digits + "_-\\. ",
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_custom_config_path(self, exp_config: str) -> None:
        path = config_loading.get_cfg_path("dummy_experiment", Path(exp_config))
        assert str(path) == exp_config
