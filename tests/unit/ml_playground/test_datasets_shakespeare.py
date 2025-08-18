from __future__ import annotations
from pytest_mock import MockerFixture
from ml_playground.datasets.shakespeare import main


def test_shakespeare_download_and_encode(mocker: MockerFixture) -> None:
    """Test shakespeare dataset downloads, splits, and encodes data."""
    test_text = "Hello world! This is test data for Shakespeare."

    # Mock components
    mock_response = mocker.Mock()
    mock_response.text = test_text
    mock_response.raise_for_status.return_value = None

    mock_encoder = mocker.Mock()
    mock_encoder.encode_ordinary.side_effect = [
        list(range(len(test_text[:42]))),  # Train data (90%)
        list(range(len(test_text[42:]))),  # Val data (10%)
    ]

    mock_path = mocker.MagicMock()
    mock_input_file = mocker.MagicMock()
    mock_train_file = mocker.MagicMock()
    mock_val_file = mocker.MagicMock()

    # Setup path mocking
    mock_path.return_value = mock_path
    mock_path.mkdir.return_value = None
    datasets_dir = mocker.MagicMock()
    ds_dir = mocker.MagicMock()

    def truediv_root(arg):
        return datasets_dir if arg == "datasets" else mocker.MagicMock()

    def truediv_datasets(arg):
        return ds_dir if arg == "shakespeare" else mocker.MagicMock()

    mock_path.__truediv__.side_effect = truediv_root
    datasets_dir.__truediv__.side_effect = truediv_datasets

    def truediv_ds(arg):
        if arg == "input.txt":
            return mock_input_file
        if arg == "train.bin":
            return mock_train_file
        if arg == "val.bin":
            return mock_val_file
        return mocker.MagicMock()

    ds_dir.__truediv__.side_effect = truediv_ds

    # File doesn't exist initially
    mock_input_file.exists.return_value = False
    mock_input_file.write_text.return_value = None
    mock_input_file.read_text.return_value = test_text

    mocker.patch("ml_playground.datasets.shakespeare.Path", return_value=mock_path)
    mocker.patch(
        "ml_playground.datasets.shakespeare.requests.get", return_value=mock_response
    )
    mocker.patch(
        "ml_playground.datasets.shakespeare.tiktoken.get_encoding",
        return_value=mock_encoder,
    )
    arr_mock = mocker.MagicMock()
    arr_mock.tobytes.return_value = b""
    mocker.patch("numpy.array", side_effect=lambda x, dtype: arr_mock)

    main()

    # Verify download happened
    mock_response.raise_for_status.assert_called_once()
    mock_input_file.write_text.assert_called_once_with(test_text, encoding="utf-8")

    # Verify encoding calls
    assert mock_encoder.encode_ordinary.call_count == 2

    # Verify file writes
    mock_train_file.write_bytes.assert_called_once()
    mock_val_file.write_bytes.assert_called_once()


def test_shakespeare_skip_download_if_exists(mocker: MockerFixture) -> None:
    """Test shakespeare dataset skips download when input file exists."""
    test_text = "Existing Shakespeare data."

    mock_encoder = mocker.Mock()
    mock_encoder.encode_ordinary.side_effect = [
        list(range(len(test_text[:23]))),  # Train data (90%)
        list(range(len(test_text[23:]))),  # Val data (10%)
    ]

    mock_path = mocker.MagicMock()
    mock_input_file = mocker.MagicMock()
    mock_train_file = mocker.MagicMock()
    mock_val_file = mocker.MagicMock()

    # Setup path mocking
    mock_path.return_value = mock_path
    mock_path.mkdir.return_value = None
    mock_path.__truediv__.side_effect = [
        mock_input_file,
        mock_train_file,
        mock_val_file,
    ]

    # File exists
    mock_input_file.exists.return_value = True
    mock_input_file.read_text.return_value = test_text

    mocker.patch("ml_playground.datasets.shakespeare.Path", return_value=mock_path)
    mock_get = mocker.patch("ml_playground.datasets.shakespeare.requests.get")
    mocker.patch(
        "ml_playground.datasets.shakespeare.tiktoken.get_encoding",
        return_value=mock_encoder,
    )
    arr_mock2 = mocker.MagicMock()
    arr_mock2.tobytes.return_value = b""
    mocker.patch("numpy.array", side_effect=lambda x, dtype: arr_mock2)

    main()

    # Verify no download
    mock_get.assert_not_called()
    mock_input_file.write_text.assert_not_called()

    # Verify processing still happens
    mock_encoder.encode_ordinary.assert_called()
    mock_train_file.write_bytes.assert_called_once()
    mock_val_file.write_bytes.assert_called_once()


def test_shakespeare_data_split_ratios(mocker: MockerFixture) -> None:
    """Test that data is split correctly into 90% train, 10% val."""
    test_text = "x" * 100  # Exactly 100 characters for easy math

    mock_encoder = mocker.Mock()
    # We'll capture what gets passed to encode_ordinary
    encoded_texts = []

    def capture_encode(text):
        encoded_texts.append(text)
        return list(range(len(text)))

    mock_encoder.encode_ordinary.side_effect = capture_encode

    mock_path = mocker.MagicMock()
    mock_input_file = mocker.MagicMock()
    mock_train_file = mocker.MagicMock()
    mock_val_file = mocker.MagicMock()

    mock_path.return_value = mock_path
    mock_path.mkdir.return_value = None
    mock_path.__truediv__.side_effect = [
        mock_input_file,
        mock_train_file,
        mock_val_file,
    ]

    mock_input_file.exists.return_value = True
    mock_input_file.read_text.return_value = test_text

    mocker.patch("ml_playground.datasets.shakespeare.Path", return_value=mock_path)
    mocker.patch(
        "ml_playground.datasets.shakespeare.tiktoken.get_encoding",
        return_value=mock_encoder,
    )
    arr_mock3 = mocker.MagicMock()
    arr_mock3.tobytes.return_value = b""
    mocker.patch("numpy.array", side_effect=lambda x, dtype: arr_mock3)

    main()

    # Should have captured train and val data
    assert len(encoded_texts) == 2
    train_text, val_text = encoded_texts

    # Verify split ratios
    assert len(train_text) == 90  # 90% of 100
    assert len(val_text) == 10  # 10% of 100
