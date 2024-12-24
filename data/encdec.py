import pickle
import tiktoken

class EncoderDecoder:
  """
  Base class that provides unified encode/decode methods.
  Handles instantiating the right subclass under self.impl.
  """

  def __init__(self, meta_path):
    """
    Initialize the correct encoding implementation in self.impl
    based on meta_path contents.
    """
    if self._is_char_encoding(meta_path):
      self.impl = CharEncoderDecoder(meta_path)
    else:
      self.impl = BPEEncoderDecoder()

  def encode(self, text):
    return self.impl.encode(text)

  def decode(self, tokens):
    return self.impl.decode(tokens)

  def _is_char_encoding(self, meta_path):
    """Check if meta_path contains a character-level encoding."""
    if meta_path is None:
      return False
    with open(meta_path, 'rb') as f:
      meta = pickle.load(f)
      return 'itos' in meta


class CharEncoderDecoder:
  """
  Encoding/decoding of text at the character level.
  Uses mappings defined in a metadata file.
  """

  def __init__(self, meta_path):
    """Load stoi/itos mappings from meta_path pickle file."""
    with open(meta_path, 'rb') as f:
      self.meta = pickle.load(f)

    self.itos = self.meta['itos']
    self.stoi = self.meta['stoi']

  def encode(self, text):
    """Encode text to a list of integers."""
    return [self.stoi[c] for c in text]

  def decode(self, tokens):
    """Decode a list of integers to text."""
    return ''.join([self.itos[i] for i in tokens])


class BPEEncoderDecoder:

  def __init__(self):
    """Create BPE encoder directly using tiktoken."""
    self.encoder = tiktoken.get_encoding("gpt2")

  def encode(self, text):
    return self.encoder.encode(text)

  def decode(self, tokens):
    return self.encoder.decode(tokens)