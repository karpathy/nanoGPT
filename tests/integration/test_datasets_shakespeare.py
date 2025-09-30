from __future__ import annotations
from pytest_mock import MockerFixture

# New strict API imports
from ml_playground.configuration.models import PreparerConfig
from ml_playground.experiments.shakespeare.preparer import ShakespearePreparer


def test_shakespeare_download_and_encode(tmp_path, mocker: MockerFixture) -> None:
    """Test shakespeare preparer downloads, splits, encodes, and writes outputs."""
    test_text = "Hello world! This is test data for Shakespeare."

    # Arrange: redirect preparer exp_dir via module __file__
    import ml_playground.experiments.shakespeare.preparer as prep_mod

    exp_dir = tmp_path / "experiments" / "shakespeare"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = exp_dir / "datasets"
    # Point module __file__ to our temp exp_dir
    monkey = mocker
    monkey.patch.object(prep_mod, "__file__", str(exp_dir / "preparer.py"))

    # Mock network and tokenizer
    mock_response = mocker.Mock()
    mock_response.text = test_text
    mock_response.raise_for_status.return_value = None
    get_mock = mocker.patch(
        "ml_playground.experiments.shakespeare.preparer.requests.get",
        return_value=mock_response,
    )

    enc_calls: list[str] = []
    mock_tok = mocker.Mock()

    def _enc(x: str):
        enc_calls.append(x)
        return list(range(len(x)))

    mock_tok.encode.side_effect = _enc
    mocker.patch.object(prep_mod, "create_tokenizer", return_value=mock_tok)

    # Spy on writer util to assert it's called
    writer_spy = mocker.patch.object(prep_mod, "write_bin_and_meta")

    # Act
    report = ShakespearePreparer().prepare(PreparerConfig())

    # Assert: download occurred and input file written
    get_mock.assert_called_once()
    assert (ds_dir / "input.txt").exists()

    # Tokenizer used twice (train/val)
    assert len(enc_calls) == 2

    # Writer called once with ds_dir
    writer_spy.assert_called_once()
    args, kwargs = writer_spy.call_args
    assert args and args[0] == ds_dir
    # Report includes created or updated files tuples
    assert hasattr(report, "created_files") and hasattr(report, "messages")


def test_shakespeare_skip_download_if_exists(tmp_path, mocker: MockerFixture) -> None:
    """Test preparer skips download when input file exists."""
    import ml_playground.experiments.shakespeare.preparer as prep_mod

    exp_dir = tmp_path / "experiments" / "shakespeare"
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "input.txt").write_text("Existing Shakespeare data.")
    mocker.patch.object(prep_mod, "__file__", str(exp_dir / "preparer.py"))

    # Patch requests.get and ensure it's NOT called
    get_mock = mocker.patch(
        "ml_playground.experiments.shakespeare.preparer.requests.get"
    )

    # Minimal tokenizer
    mock_tok = mocker.Mock()
    mock_tok.encode.side_effect = lambda s: list(range(len(s)))
    mocker.patch.object(prep_mod, "create_tokenizer", return_value=mock_tok)
    writer_spy = mocker.patch.object(prep_mod, "write_bin_and_meta")

    ShakespearePreparer().prepare(PreparerConfig())

    get_mock.assert_not_called()
    writer_spy.assert_called_once()
    assert mock_tok.encode.call_count == 2


def test_shakespeare_data_split_ratios(tmp_path, mocker: MockerFixture) -> None:
    """Test that data is split into 90% train, 10% val before encoding."""
    import ml_playground.experiments.shakespeare.preparer as prep_mod

    exp_dir = tmp_path / "experiments" / "shakespeare"
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    test_text = "x" * 100
    (ds_dir / "input.txt").write_text(test_text)
    mocker.patch.object(prep_mod, "__file__", str(exp_dir / "preparer.py"))

    captured: list[str] = []
    mock_tok = mocker.Mock()

    def _enc(s: str):
        captured.append(s)
        return list(range(len(s)))

    mock_tok.encode.side_effect = _enc
    mocker.patch.object(prep_mod, "create_tokenizer", return_value=mock_tok)
    mocker.patch.object(prep_mod, "write_bin_and_meta")

    ShakespearePreparer().prepare(PreparerConfig())

    assert len(captured) == 2
    train_text, val_text = captured
    assert len(train_text) == 90
    assert len(val_text) == 10
