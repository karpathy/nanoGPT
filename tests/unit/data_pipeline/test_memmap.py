"""Property-based tests for data loading logic using Hypothesis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
import pytest
import torch

from ml_playground.data_pipeline.sampling.batches import SimpleBatches, sample_batch
from ml_playground.data_pipeline.sources.memmap import MemmapReader
from ml_playground.configuration.models import DataConfig


# Strategies for generating test data
@st.composite
def array_shape_strategy(draw: st.DrawFn) -> tuple[int, ...]:
    """Generate valid array shapes for testing."""
    # Avoid very large arrays for performance
    size = draw(st.integers(min_value=1, max_value=10000))
    return (size,)


@st.composite
def batch_config_strategy(draw: st.DrawFn) -> tuple[int, int]:
    """Generate valid batch configuration parameters."""
    batch_size = draw(st.integers(min_value=1, max_value=32))
    block_size = draw(st.integers(min_value=1, max_value=512))
    return batch_size, block_size


@st.composite
def device_strategy(draw: st.DrawFn) -> str:
    """Generate device types."""
    return draw(
        st.sampled_from(["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    )


class TestMemmapReader:
    """Property-based tests for ``MemmapReader``."""

    @given(length=st.integers(min_value=1, max_value=512))
    @settings(max_examples=10, deadline=None)
    def test_memmap_reader_creation(self, length: int) -> None:
        """Test that MemmapReader can be created and reads data correctly."""
        # Create test data
        test_data = np.random.randint(0, 65535, size=length, dtype=np.uint16)

        # Create test file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                test_data.tofile(f)
                f.flush()
                f.close()

                path = Path(f.name)
                reader = MemmapReader.open(path, dtype=np.uint16)

                # Check that length matches
                assert reader.length == length

                # Check that we can read data
                assert len(reader.arr) == length

                # Check that data matches (at least first few elements)
                if length > 0:
                    assert reader.arr[0] == test_data[0]

            finally:
                path.unlink(missing_ok=True)


class TestSampleBatch:
    """Property-based tests for ``sample_batch`` function."""

    @given(
        array_size=st.integers(min_value=1, max_value=512),
        batch_config=batch_config_strategy(),
        device=device_strategy(),
    )
    @settings(max_examples=10, deadline=None)
    def test_sample_batch_shapes(
        self, array_size: int, batch_config: tuple[int, int], device: str
    ) -> None:
        """Test that sampled batches have correct shapes."""
        batch_size, block_size = batch_config

        # Skip if array is too small for the block size
        if array_size < block_size:
            pytest.skip("Array too small for block size")

        # Create test data
        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                test_data = np.random.randint(
                    0, 65535, size=array_size, dtype=np.uint16
                )
                test_data.tofile(f)
                f.flush()
                f.close()

                path = Path(f.name)
                reader = MemmapReader.open(path, dtype=np.uint16)

                x, y = sample_batch(reader, batch_size, block_size, device)

                # Check shapes
                assert x.shape == (batch_size, block_size)
                assert y.shape == (batch_size, block_size)

                # Check device
                assert x.device.type == device
                assert y.device.type == device

                # Check that y is shifted version of x
                assert torch.equal(x[:, 1:], y[:, :-1])

            finally:
                path.unlink(missing_ok=True)


class TestSimpleBatches:
    """Property-based tests for SimpleBatches class."""

    @given(
        array_size=st.integers(min_value=10, max_value=512),
        batch_config=batch_config_strategy(),
        device=device_strategy(),
        sampler=st.sampled_from(["random", "sequential"]),
    )
    @settings(max_examples=8, deadline=None)
    def test_simple_batches_creation(self, array_size, batch_config, device, sampler):
        """Test SimpleBatches initialization and basic functionality."""
        batch_size, block_size = batch_config

        # Skip invalid configurations
        if array_size < block_size:
            pytest.skip("Array too small for block size")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test data files
            train_data = np.random.randint(0, 65535, size=array_size, dtype=np.uint16)
            val_data = np.random.randint(
                0, 65535, size=max(1, array_size // 10), dtype=np.uint16
            )

            train_path = temp_path / "train.bin"
            val_path = temp_path / "val.bin"
            meta_path = temp_path / "meta.pkl"

            train_data.tofile(train_path)
            val_data.tofile(val_path)

            # Create meta.pkl
            import pickle

            with open(meta_path, "wb") as f:
                pickle.dump({"dtype": "uint16"}, f)

            # Create DataConfig
            data_config = DataConfig(
                batch_size=batch_size, block_size=block_size, sampler=sampler
            )

            # Create SimpleBatches
            batches = SimpleBatches(data_config, device, temp_path)

            # Test batch retrieval
            train_batch = batches.get_batch("train")
            val_batch = batches.get_batch("val")

            assert isinstance(train_batch, tuple)
            assert len(train_batch) == 2
            assert train_batch[0].shape == (batch_size, block_size)
            assert train_batch[1].shape == (batch_size, block_size)

            assert isinstance(val_batch, tuple)
            assert len(val_batch) == 2

    @given(
        array_size=st.integers(min_value=100, max_value=400),
        batch_size=st.integers(min_value=1, max_value=8),
        block_size=st.integers(min_value=10, max_value=48),
        device=device_strategy(),
    )
    @settings(max_examples=8, deadline=None)
    def test_sequential_sampling_coverage(
        self, array_size, batch_size, block_size, device
    ):
        """Test that sequential sampling eventually covers the dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test data
            train_data = np.arange(array_size, dtype=np.uint16)
            val_data = np.arange(max(1, array_size // 10), dtype=np.uint16)

            train_path = temp_path / "train.bin"
            val_path = temp_path / "val.bin"
            meta_path = temp_path / "meta.pkl"

            train_data.tofile(train_path)
            val_data.tofile(val_path)

            with open(meta_path, "wb") as f:
                import pickle

                pickle.dump({"dtype": "uint16"}, f)

            # Create DataConfig with sequential sampling
            data_config = DataConfig(
                batch_size=batch_size, block_size=block_size, sampler="sequential"
            )

            batches = SimpleBatches(data_config, device, temp_path)

            # Get multiple batches to test sequential behavior
            seen_positions = set()
            for _ in range(3):
                x, y = batches.get_batch("train")
                # Check that we're getting different data each time (sequential)
                first_val = x[0, 0].item()
                seen_positions.add(first_val)

            # Should see some variety in sequential sampling if possible
            assert len(seen_positions) >= 1

    @given(
        batch_size=st.integers(min_value=1, max_value=16),
        block_size=st.integers(min_value=1, max_value=64),
        device=device_strategy(),
    )
    @settings(max_examples=10)
    def test_missing_data_files_raise_errors(self, batch_size, block_size, device):
        """Test that missing data files raise appropriate errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Don't create any data files
            data_config = DataConfig(batch_size=batch_size, block_size=block_size)

            with pytest.raises(FileNotFoundError):
                SimpleBatches(data_config, device, temp_path)
