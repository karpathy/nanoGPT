from __future__ import annotations

import json
import math
import os
import random
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal, cast, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import TYPE_CHECKING

# Add TensorBoard (best-effort)
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:
    SummaryWriter = None  # type: ignore

# Add PEFT (best-effort)
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError:
    LoraConfig = get_peft_model = TaskType = PeftModel = None  # type: ignore

if TYPE_CHECKING:
    pass


# -----------------------------
# Configuration dataclasses
# -----------------------------


@dataclass(frozen=True)
class PrepareCfg:
    """Configuration for SpeakGer dataset preparation."""
    raw_dir: Path
    dataset_dir: Path
    add_structure_tokens: bool = True
    doc_separator: str = "<DOC_SEP>"


@dataclass(frozen=True)
class HFModelCfg:
    """Hugging Face model configuration for Gemma 3."""
    model_name: str = "google/gemma-2-2b"  # Default to 2B, can be switched to 270M or 7B
    gradient_checkpointing: bool = True
    block_size: int = 512  # Conservative for 32GB RAM


@dataclass(frozen=True)
class PeftCfg:
    """LoRA/PEFT configuration for efficient fine-tuning."""
    enabled: bool = True
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "o_proj")
    extend_mlp_targets: bool = False


@dataclass(frozen=True)
class DataCfg:
    """Data loading configuration."""
    dataset_dir: Path
    batch_size: int = 6  # Conservative for 32GB RAM + MPS
    grad_accum_steps: int = 10
    block_size: int = 512
    shuffle: bool = True


@dataclass(frozen=True)
class OptimCfg:
    """Optimizer configuration."""
    learning_rate: float = 0.0015  # Slightly lower for Gemma
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass(frozen=True)
class SchedCfg:
    """Learning rate schedule configuration."""
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int = 10000
    min_lr: float = 0.00015


@dataclass(frozen=True)
class RuntimeCfg:
    """Runtime configuration for training."""
    out_dir: Path
    max_iters: int = 10000  # Conservative for local training
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 25
    eval_only: bool = False
    always_save_checkpoint: bool = True
    seed: int = 1337
    device: str = "mps"
    dtype: str = "float16"
    compile: bool = False
    ckpt_time_interval_minutes: int = 15
    ckpt_atomic: bool = True
    best_smoothing_alpha: float = 0.0
    early_stop_patience: int = 3
    ema_decay: float = 0.0
    save_merged_on_best: bool = True
    keep_last_n: int = 2


@dataclass(frozen=True)
class SampleRuntime:
    """Runtime configuration for sampling."""
    out_dir: Path
    device: str = "mps"
    dtype: str = "float16"
    compile: bool = False
    seed: int = 1337


@dataclass(frozen=True)
class SampleCfg:
    """Sampling configuration."""
    start: str = "\nSprecher: Dr. Alice Weidel (AfD)\nThema: Aktuelle politische Entwicklungen\nJahr: 2024\n\n"
    num_samples: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_k: int = 200
    top_p: float = 0.9


@dataclass(frozen=True)
class TrainTOML:
    hf_model: HFModelCfg
    peft: PeftCfg
    data: DataCfg
    optim: OptimCfg
    schedule: SchedCfg
    runtime: RuntimeCfg


@dataclass(frozen=True)
class SampleTOML:
    runtime: SampleRuntime
    sample: SampleCfg


@dataclass(frozen=True)
class AppTOML:
    prepare: PrepareCfg
    train: TrainTOML
    sample: SampleTOML


# -----------------------------
# Helpers
# -----------------------------


def _read_toml(config_path: Path) -> dict:
    import tomllib

    with config_path.open("rb") as f:
        return tomllib.load(f)


def _coerce_path(d: dict, key: str) -> Path:
    return Path(d[key])


def _parse_bias(val: object) -> Literal["none", "all", "lora_only"]:
    s = str(val)
    if s not in ("none", "all", "lora_only"):
        s = "none"
    return cast(Literal["none", "all", "lora_only"], s)


def _parse_app(config_path: Path) -> AppTOML:
    d = _read_toml(config_path)
    p = d.get("prepare", {})
    t = d.get("train", {})
    s = d.get("sample", {})

    prepare = PrepareCfg(
        raw_dir=_coerce_path(p, "raw_dir"),
        dataset_dir=_coerce_path(p, "dataset_dir"),
        add_structure_tokens=bool(p.get("add_structure_tokens", True)),
        doc_separator=str(p.get("doc_separator", "<DOC_SEP>")),
    )
    hf_model = HFModelCfg(**t.get("hf_model", {}))
    peft_cfg = PeftCfg(
        enabled=bool(t.get("peft", {}).get("enabled", True)),
        r=int(t.get("peft", {}).get("r", 8)),
        lora_alpha=int(t.get("peft", {}).get("lora_alpha", 16)),
        lora_dropout=float(t.get("peft", {}).get("lora_dropout", 0.05)),
        bias=_parse_bias(t.get("peft", {}).get("bias", "none")),
        target_modules=tuple(
            t.get("peft", {}).get("target_modules", ["q_proj", "v_proj", "o_proj"])
        ),
        extend_mlp_targets=bool(t.get("peft", {}).get("extend_mlp_targets", False)),
    )
    data_cfg = DataCfg(
        dataset_dir=_coerce_path(t.get("data", {}), "dataset_dir"),
        batch_size=int(t.get("data", {}).get("batch_size", 6)),
        grad_accum_steps=int(t.get("data", {}).get("grad_accum_steps", 10)),
        block_size=int(t.get("data", {}).get("block_size", hf_model.block_size)),
        shuffle=bool(t.get("data", {}).get("shuffle", True)),
    )
    optim_cfg = OptimCfg(**t.get("optim", {}))
    sched_cfg = SchedCfg(**t.get("schedule", {}))
    runtime_cfg = RuntimeCfg(
        out_dir=_coerce_path(t.get("runtime", {}), "out_dir"),
        max_iters=int(t.get("runtime", {}).get("max_iters", 10_000)),
        eval_interval=int(t.get("runtime", {}).get("eval_interval", 250)),
        eval_iters=int(t.get("runtime", {}).get("eval_iters", 50)),
        log_interval=int(t.get("runtime", {}).get("log_interval", 25)),
        eval_only=bool(t.get("runtime", {}).get("eval_only", False)),
        always_save_checkpoint=bool(
            t.get("runtime", {}).get("always_save_checkpoint", True)
        ),
        seed=int(t.get("runtime", {}).get("seed", 1337)),
        device=str(t.get("runtime", {}).get("device", "mps")),
        dtype=str(t.get("runtime", {}).get("dtype", "float16")),
        compile=bool(t.get("runtime", {}).get("compile", False)),
        ckpt_time_interval_minutes=int(
            t.get("runtime", {}).get("ckpt_time_interval_minutes", 15)
        ),
        ckpt_atomic=bool(t.get("runtime", {}).get("ckpt_atomic", True)),
        best_smoothing_alpha=float(
            t.get("runtime", {}).get("best_smoothing_alpha", 0.0)
        ),
        early_stop_patience=int(t.get("runtime", {}).get("early_stop_patience", 3)),
        ema_decay=float(t.get("runtime", {}).get("ema_decay", 0.0)),
        save_merged_on_best=bool(
            t.get("runtime", {}).get("save_merged_on_best", True)
        ),
        keep_last_n=int(t.get("runtime", {}).get("keep_last_n", 2)),
    )

    sample_runtime = SampleRuntime(
        out_dir=_coerce_path(s.get("runtime", {}), "out_dir"),
        device=str(s.get("runtime", {}).get("device", "mps")),
        dtype=str(s.get("runtime", {}).get("dtype", "float16")),
        compile=bool(s.get("runtime", {}).get("compile", False)),
        seed=int(s.get("runtime", {}).get("seed", 1337)),
    )
    sample_cfg = SampleCfg(**s.get("sample", {}))

    return AppTOML(
        prepare=prepare,
        train=TrainTOML(
            hf_model=hf_model,
            peft=peft_cfg,
            data=data_cfg,
            optim=optim_cfg,
            schedule=sched_cfg,
            runtime=runtime_cfg,
        ),
        sample=SampleTOML(runtime=sample_runtime, sample=sample_cfg),
    )


# -----------------------------
# SpeakGer Dataset utilities
# -----------------------------


def _read_all_texts(raw_dir: Path) -> List[str]:
    """Read all text files from raw_dir recursively."""
    texts = []
    for file_path in raw_dir.rglob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    return texts


def _decorate_texts(texts: List[str], add_tokens: bool) -> List[str]:
    """Add structure tokens to texts if requested."""
    if not add_tokens:
        return texts
    
    # Simple structure decoration for SpeakGer format
    decorated = []
    for text in texts:
        decorated_text = f"<SPEECH_START>{text}<SPEECH_END>"
        decorated.append(decorated_text)
    return decorated


def _extract_fields(row: dict[str, str]) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract content and common metadata fields from a CSV row.

    Returns: (content, speaker, party, year, topic)
    """
    # Try common content field names
    content_keys = [
        "text",
        "speech",
        "content",
        "body",
        "rede",
        "speech_text",
        "transcript",
        "full_text",
        "speechContent",
    ]
    content = None
    for k in content_keys:
        if k in row and row[k]:
            content = row[k]
            break
    if content is None:
        # Fallback: join all non-empty fields as content
        content = " ".join(v for v in row.values() if v)

    # Speaker
    speaker_keys = [
        "speaker",
        "redner",
        "name",
        "speaker_name",
        "PersonName",
        "person",
        "speaker_fullname",
        "firstname",
        "lastname",
    ]
    speaker = None
    first = row.get("firstname") or row.get("first_name")
    last = row.get("lastname") or row.get("last_name")
    if first or last:
        speaker = " ".join(x for x in [first, last] if x)
    if not speaker:
        for k in speaker_keys:
            if k in row and row[k]:
                speaker = row[k]
                break

    # Party
    party_keys = [
        "party",
        "partei",
        "fraction",
        "faction",
        "parliamentary_group",
        "Fraktion",
        "party_long",
        "party_short",
    ]
    party = None
    for k in party_keys:
        if k in row and row[k]:
            party = row[k]
            break

    # Year (try date-like fields)
    import re

    year = None
    date_keys = [
        "date",
        "datum",
        "year",
        "jahr",
        "time",
        "timestamp",
        "session_date",
        "speech_date",
    ]
    for k in date_keys:
        if k in row and row[k]:
            m = re.search(r"(19|20)\d{2}", row[k])
            if m:
                year = m.group(0)
                break

    # Topic/Title
    topic_keys = [
        "topic",
        "title",
        "subject",
        "agenda_item",
        "thema",
        "heading",
    ]
    topic = None
    for k in topic_keys:
        if k in row and row[k]:
            topic = row[k]
            break

    return content, speaker, party, year, topic


def _format_conditioned_text(content: str, speaker: Optional[str], party: Optional[str], year: Optional[str], topic: Optional[str], add_structure_tokens: bool) -> str:
    """Build a single training example string with optional metadata headers."""
    headers = []
    if speaker:
        headers.append(f"Sprecher: {speaker}")
    if party:
        headers.append(f"Partei: {party}")
    if year:
        headers.append(f"Jahr: {year}")
    if topic:
        headers.append(f"Thema: {topic}")

    header_txt = "\n".join(headers)
    body = content.strip()

    if add_structure_tokens:
        return f"<SPEECH_START>\n{header_txt}\n\n{body}\n<SPEECH_END>" if header_txt else f"<SPEECH_START>\n{body}\n<SPEECH_END>"
    else:
        return f"{header_txt}\n\n{body}" if header_txt else body


def _prepare_from_csv(csv_path: Path, out_dir: Path, add_structure_tokens: bool) -> dict[str, int]:
    """Stream a large CSV and create train/val JSONL with ~90/10 split.

    Returns meta counts.
    """
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    total = train_n = val_n = 0

    # Use newline='' for csv module per docs
    with open(csv_path, "r", encoding="utf-8", newline="") as f_in, \
         open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"CSV appears to have no header: {csv_path}")
        for row in reader:
            total += 1
            try:
                content, speaker, party, year, topic = _extract_fields(row)
                text = _format_conditioned_text(content, speaker, party, year, topic, add_structure_tokens)
                rec = {"text": text}
                # Probabilistic split ~10% to val
                if random.random() < 0.10:
                    json.dump(rec, f_val, ensure_ascii=False)
                    f_val.write("\n")
                    val_n += 1
                else:
                    json.dump(rec, f_train, ensure_ascii=False)
                    f_train.write("\n")
                    train_n += 1
            except Exception:
                # Skip malformed rows silently but continue
                continue

    meta = {
        "total_texts": total,
        "train_texts": train_n,
        "val_texts": val_n,
        "source": str(csv_path),
        "from_csv": True,
        "add_structure_tokens": add_structure_tokens,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=2, ensure_ascii=False)
    return meta


def prepare_from_toml(config_path: Path) -> None:
    """Prepare SpeakGer dataset from TOML configuration."""
    cfg = _parse_app(config_path)
    prepare_cfg = cfg.prepare

    print(f"[gemma_finetuning_mps] Preparing dataset from {prepare_cfg.raw_dir}")

    # Create dataset directory
    prepare_cfg.dataset_dir.mkdir(parents=True, exist_ok=True)

    # Branch: CSV vs text directory
    if prepare_cfg.raw_dir.is_file() and prepare_cfg.raw_dir.suffix.lower() == ".csv":
        meta = _prepare_from_csv(prepare_cfg.raw_dir, prepare_cfg.dataset_dir, prepare_cfg.add_structure_tokens)
        print(
            f"[gemma_finetuning_mps] Prepared CSV dataset: {meta['train_texts']} train, {meta['val_texts']} val (total {meta['total_texts']})"
        )
        return

    if not prepare_cfg.raw_dir.exists() or not prepare_cfg.raw_dir.is_dir():
        raise ValueError(f"raw_dir must be a directory of .txt files or a .csv file: {prepare_cfg.raw_dir}")

    # Read all text files
    texts = _read_all_texts(prepare_cfg.raw_dir)
    if not texts:
        raise ValueError(f"No text files found in {prepare_cfg.raw_dir}")

    print(f"[gemma_finetuning_mps] Found {len(texts)} text files")

    # Add structure tokens if requested
    texts = _decorate_texts(texts, prepare_cfg.add_structure_tokens)

    # Split into train/val (90/10)
    random.shuffle(texts)
    split_idx = int(0.9 * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    # Write JSONL files
    train_jsonl = prepare_cfg.dataset_dir / "train.jsonl"
    val_jsonl = prepare_cfg.dataset_dir / "val.jsonl"

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for text in train_texts:
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")

    with open(val_jsonl, "w", encoding="utf-8") as f:
        for text in val_texts:
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")

    # Write metadata
    meta = {
        "total_texts": len(texts),
        "train_texts": len(train_texts),
        "val_texts": len(val_texts),
        "add_structure_tokens": prepare_cfg.add_structure_tokens,
        "doc_separator": prepare_cfg.doc_separator,
        "from_csv": False,
    }

    with open(prepare_cfg.dataset_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[gemma_finetuning_mps] Prepared dataset: {len(train_texts)} train, {len(val_texts)} val")


class PackedJSONL(Dataset):
    """Dataset for loading packed JSONL data."""
    
    def __init__(self, jsonl_path: Path, tokenizer, block_size: int, sep_token: Optional[str] = None):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.sep_token = sep_token
        
        # Load and pack all texts
        texts = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                texts.append(data["text"])
        
        # Join all texts with separator
        if sep_token and sep_token in tokenizer.get_vocab():
            full_text = f" {sep_token} ".join(texts)
        else:
            full_text = "\n\n".join(texts)
        
        # Tokenize the full text
        tokens = tokenizer.encode(full_text)
        
        # Split into blocks
        self.blocks = []
        for i in range(0, len(tokens) - block_size + 1, block_size):
            block = tokens[i:i + block_size]
            self.blocks.append(torch.tensor(block, dtype=torch.long))
    
    def __len__(self) -> int:
        return len(self.blocks)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.blocks[idx]


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _autocast_ctx(device_str: str, dtype: torch.dtype):
    """Create autocast context for the given device."""
    if device_str == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    elif device_str == "mps":
        return torch.cpu.amp.autocast(dtype=dtype)
    else:
        return torch.cpu.amp.autocast(dtype=dtype)


def _hf_cache_dir() -> Path:
    """Return a persistent cache directory for HF/Transformers downloads.

    Resolution order (first present wins):
    - TRANSFORMERS_CACHE
    - HUGGINGFACE_HUB_CACHE
    - HF_HOME
    - Project-local ".hf_cache" at repository root (default)
    """
    for key in ("TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        val = os.environ.get(key)
        if val and str(val).strip():
            p = Path(val).expanduser()
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                # best-effort
                pass
            return p
    # fallback to project-local cache
    repo_root = Path(__file__).resolve().parent.parent.parent
    default = repo_root / ".hf_cache"
    try:
        default.mkdir(parents=True, exist_ok=True)
    except Exception:
        # best-effort
        pass
    return default


def _get_hf_token() -> Optional[str]:
    """Return an HF token from env or fallback .env files when needed.

    Order of resolution:
    1) Existing non-empty environment variables (HUGGINGFACE_HUB_TOKEN, HUGGINGFACEHUB_API_TOKEN, HF_TOKEN)
    2) .env files in CWD and repo root (parsed best-effort) if no non-empty env var found or if the present one is empty.
    """
    # 1) Check existing environment first (non-empty only)
    for key in ("HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"):
        tok = os.environ.get(key)
        if tok and tok.strip():
            return tok.strip()

    # 2) Fallback: try to parse .env files from CWD and repo root best-effort
    def _parse_env_file(path: Path) -> None:
        try:
            if not path.exists():
                return
            for line in path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # Only set if currently unset or empty
                if k in ("HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"):
                    if not os.environ.get(k) or not os.environ.get(k, "").strip():
                        os.environ[k] = v
        except Exception:
            # Best-effort only
            pass

    # CWD .env
    _parse_env_file(Path.cwd() / ".env")
    # Repo root .env: this file is at ml_playground/datasets/*.py → repo root is three parents up
    repo_root = Path(__file__).resolve().parent.parent.parent
    _parse_env_file(repo_root / ".env")

    # Re-check after parsing
    for key in ("HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"):
        tok = os.environ.get(key)
        if tok and tok.strip():
            return tok.strip()

    return None


def _maybe_hf_login() -> Optional[str]:
    """If a token is present in env, perform a non-interactive login to cache it.

    Returns the token if available, else None.
    """
    token = _get_hf_token()
    if not token:
        return None
    try:
        # Import lazily to avoid hard dependency at import time.
        from huggingface_hub import login  # type: ignore

        login(token=token, add_to_git_credential=False)
    except Exception:
        # Best-effort only; downstream calls will still receive the token directly.
        pass
    return token


def _auth_kwargs() -> dict[str, Any]:
    """Build keyword args for from_pretrained with auth token when available.

    We pass both 'token' and legacy 'use_auth_token' for compatibility across
    transformers/huggingface_hub versions.
    """
    token = _maybe_hf_login()
    if not token:
        return {}
    return {"token": token, "use_auth_token": token}


def _raise_gated_help(exc: Exception, model_name: str) -> None:
    """Raise a clearer error with instructions if access to a gated repo fails."""
    msg = str(exc)
    lower = msg.lower()
    if ("gated" in lower) or ("401" in lower) or ("unauthorized" in lower):
        help_msg = (
            f"\n[gemma_finetuning_mps] Authentication required for gated model '{model_name}'.\n"
            "Do one of the following (UV-only):\n"
            "  1) Export token just for this shell: export HUGGINGFACE_HUB_TOKEN=hf_...\n"
            "  2) Persist login in the venv: uv run huggingface-cli login --token hf_...\n"
            "     or: uv run python -c \"from huggingface_hub import login; login(token='hf_...')\"\n"
            "Then re-run your ml_playground command.\n"
        )
        raise OSError(msg + help_msg) from exc
    raise exc


def _save_adapters(model: nn.Module, save_dir: Path, atomic: bool = False) -> None:
    """Save PEFT adapters."""
    if atomic:
        temp_dir = save_dir.with_suffix(".tmp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(temp_dir))
        if save_dir.exists():
            shutil.rmtree(save_dir)
        temp_dir.rename(save_dir)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))


def train_from_toml(config_path: Path) -> None:
    """Train Gemma model from TOML configuration."""
    if LoraConfig is None:
        raise ImportError("PEFT is required for fine-tuning. Install with: uv add peft")
    
    cfg = _parse_app(config_path)
    train_cfg = cfg.train
    
    print(f"[gemma_finetuning_mps] Starting training with config: {config_path}")
    print(f"[gemma_finetuning_mps] Model: {train_cfg.hf_model.model_name}")
    print(f"[gemma_finetuning_mps] Device: {train_cfg.runtime.device}")
    
    # Set seed
    _set_seed(train_cfg.runtime.seed)
    
    # Setup device and dtype
    device = torch.device(train_cfg.runtime.device)
    dtype = getattr(torch, train_cfg.runtime.dtype)
    
    # Create output directory
    train_cfg.runtime.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    tb_writer = None
    if SummaryWriter is not None:
        tb_dir = train_cfg.runtime.out_dir / "logs" / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_dir))
    
    # Load tokenizer and model
    print("[gemma_finetuning_mps] Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            train_cfg.hf_model.model_name,
            cache_dir=_hf_cache_dir(),
            **_auth_kwargs(),
        )
    except Exception as e:
        _raise_gated_help(e, train_cfg.hf_model.model_name)
        raise
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            train_cfg.hf_model.model_name,
            torch_dtype=dtype,
            device_map={"": device} if device.type != "cpu" else None,
            cache_dir=_hf_cache_dir(),
            **_auth_kwargs(),
        )
    except Exception as e:
        _raise_gated_help(e, train_cfg.hf_model.model_name)
        raise
    
    if train_cfg.hf_model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Setup PEFT
    if train_cfg.peft.enabled:
        print("[gemma_finetuning_mps] Setting up LoRA...")
        target_modules = list(train_cfg.peft.target_modules)
        if train_cfg.peft.extend_mlp_targets:
            target_modules.extend(["up_proj", "down_proj", "gate_proj"])
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=train_cfg.peft.r,
            lora_alpha=train_cfg.peft.lora_alpha,
            lora_dropout=train_cfg.peft.lora_dropout,
            bias=train_cfg.peft.bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset = PackedJSONL(
        train_cfg.data.dataset_dir / "train.jsonl",
        tokenizer,
        train_cfg.data.block_size,
        cfg.prepare.doc_separator if cfg.prepare.doc_separator != "<DOC_SEP>" else None,
    )
    
    val_dataset = PackedJSONL(
        train_cfg.data.dataset_dir / "val.jsonl",
        tokenizer,
        train_cfg.data.block_size,
        cfg.prepare.doc_separator if cfg.prepare.doc_separator != "<DOC_SEP>" else None,
    )
    
    print(f"[gemma_finetuning_mps] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.data.batch_size,
        shuffle=train_cfg.data.shuffle,
        pin_memory=device.type == "cuda",
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.data.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.optim.learning_rate,
        weight_decay=train_cfg.optim.weight_decay,
        betas=(train_cfg.optim.beta1, train_cfg.optim.beta2),
    )
    
    # Setup scheduler
    if train_cfg.schedule.decay_lr:
        def get_lr(it):
            if it < train_cfg.schedule.warmup_iters:
                return train_cfg.optim.learning_rate * it / train_cfg.schedule.warmup_iters
            if it > train_cfg.schedule.lr_decay_iters:
                return train_cfg.schedule.min_lr
            decay_ratio = (it - train_cfg.schedule.warmup_iters) / (train_cfg.schedule.lr_decay_iters - train_cfg.schedule.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return train_cfg.schedule.min_lr + coeff * (train_cfg.optim.learning_rate - train_cfg.schedule.min_lr)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    else:
        scheduler = None
    
    # Training state
    model.train()
    iter_num = 0
    best_val_loss = float('inf')
    step_count = 0
    tokens_per_sec = 0.0
    last_time = time.time()
    
    print(f"[gemma_finetuning_mps] Starting training for {train_cfg.runtime.max_iters} iterations...")
    
    while iter_num < train_cfg.runtime.max_iters:
        for batch in train_loader:
            if iter_num >= train_cfg.runtime.max_iters:
                break
            
            # Move batch to device
            inputs = batch.to(device)
            targets = inputs.clone()
            
            # Forward pass with autocast
            with _autocast_ctx(train_cfg.runtime.device, dtype):
                outputs = model(inputs, labels=targets)
                loss = outputs.loss / train_cfg.data.grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (iter_num + 1) % train_cfg.data.grad_accum_steps == 0:
                if train_cfg.optim.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.optim.grad_clip)
                
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                
                # Calculate tokens per second
                now = time.time()
                dt = now - last_time
                if dt > 0:
                    tokens_per_sec = (train_cfg.data.batch_size * train_cfg.data.block_size * train_cfg.data.grad_accum_steps) / dt
                last_time = now
            
            # Logging
            if iter_num % train_cfg.runtime.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"iter {iter_num:6d}: loss {loss.item():.4f}, lr {lr:.2e}, tokens/sec {tokens_per_sec:.1f}")
                
                if tb_writer is not None:
                    tb_writer.add_scalar('train/loss', loss.item(), iter_num)
                    tb_writer.add_scalar('train/lr', lr, iter_num)
                    tb_writer.add_scalar('train/tokens_per_sec', tokens_per_sec, iter_num)
            
            # Evaluation
            if iter_num % train_cfg.runtime.eval_interval == 0 and iter_num > 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        if val_steps >= train_cfg.runtime.eval_iters:
                            break
                        
                        val_inputs = val_batch.to(device)
                        val_targets = val_inputs.clone()
                        
                        with _autocast_ctx(train_cfg.runtime.device, dtype):
                            val_outputs = model(val_inputs, labels=val_targets)
                            val_loss += val_outputs.loss.item()
                        
                        val_steps += 1
                
                val_loss /= val_steps
                print(f"iter {iter_num:6d}: val loss {val_loss:.4f}")
                
                if tb_writer is not None:
                    tb_writer.add_scalar('val/loss', val_loss, iter_num)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}")
                    
                    if train_cfg.peft.enabled:
                        best_dir = train_cfg.runtime.out_dir / "adapters" / "best"
                        _save_adapters(model, best_dir, train_cfg.runtime.ckpt_atomic)
                
                # Save last checkpoint
                if train_cfg.runtime.always_save_checkpoint:
                    if train_cfg.peft.enabled:
                        last_dir = train_cfg.runtime.out_dir / "adapters" / "last"
                        _save_adapters(model, last_dir, train_cfg.runtime.ckpt_atomic)
                
                model.train()
            
            iter_num += 1
    
    print(f"[gemma_finetuning_mps] Training completed after {iter_num} iterations")
    
    # Final save
    if train_cfg.peft.enabled:
        final_dir = train_cfg.runtime.out_dir / "adapters" / "final"
        _save_adapters(model, final_dir, train_cfg.runtime.ckpt_atomic)
        
        # Save tokenizer
        tokenizer.save_pretrained(str(train_cfg.runtime.out_dir / "tokenizer"))
    
    if tb_writer is not None:
        tb_writer.close()


def sample_from_toml(config_path: Path) -> None:
    """Generate samples from trained Gemma model."""
    if PeftModel is None:
        raise ImportError("PEFT is required for sampling. Install with: uv add peft")
    
    cfg = _parse_app(config_path)
    sample_cfg = cfg.sample
    
    print(f"[gemma_finetuning_mps] Starting sampling with config: {config_path}")
    
    # Set seed
    _set_seed(sample_cfg.runtime.seed)
    
    # Setup device and dtype
    device = torch.device(sample_cfg.runtime.device)
    dtype = getattr(torch, sample_cfg.runtime.dtype)
    
    # Find adapters directory
    adapters_dir = None
    for subdir_name in ["best", "final", "last"]:
        potential_dir = sample_cfg.runtime.out_dir / "adapters" / subdir_name
        if potential_dir.exists():
            adapters_dir = potential_dir
            print(f"[gemma_finetuning_mps] Using adapters from: {adapters_dir}")
            break
    
    if adapters_dir is None:
        raise FileNotFoundError(f"No adapters found in {sample_cfg.runtime.out_dir / 'adapters'}")
    
    # Load tokenizer
    tokenizer_dir = sample_cfg.runtime.out_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    else:
        # Fallback to base model tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.train.hf_model.model_name,
                cache_dir=_hf_cache_dir(),
                **_auth_kwargs(),
            )
        except Exception as e:
            _raise_gated_help(e, cfg.train.hf_model.model_name)
            raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model and adapters
    print("[gemma_finetuning_mps] Loading base model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.train.hf_model.model_name,
            torch_dtype=dtype,
            device_map={"": device} if device.type != "cpu" else None,
            cache_dir=_hf_cache_dir(),
            **_auth_kwargs(),
        )
    except Exception as e:
        _raise_gated_help(e, cfg.train.hf_model.model_name)
        raise
    
    print("[gemma_finetuning_mps] Loading fine-tuned adapters...")
    model = PeftModel.from_pretrained(base_model, str(adapters_dir))
    
    if sample_cfg.runtime.compile:
        model = torch.compile(model)
    
    model.eval()
    
    # Generate samples
    for i in range(sample_cfg.sample.num_samples):
        print(f"\n[gemma_finetuning_mps] Generating sample {i+1}/{sample_cfg.sample.num_samples}...")
        
        # Tokenize prompt
        prompt = sample_cfg.sample.start
        if prompt.startswith("FILE:"):
            prompt_path = Path(prompt[5:])
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=sample_cfg.sample.max_new_tokens,
                temperature=sample_cfg.sample.temperature,
                top_k=sample_cfg.sample.top_k,
                top_p=sample_cfg.sample.top_p,
                do_sample=sample_cfg.sample.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and save
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Create samples directory
        samples_dir = sample_cfg.runtime.out_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Save sample
        timestamp = int(time.time())
        sample_file = samples_dir / f"sample-{timestamp}.txt"
        
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write("=" * 50 + "\n")
            f.write(generated_text)
            f.write("\n")
        
        print(f"[gemma_finetuning_mps] Sample saved to: {sample_file}")
        
        # Also print to console (truncated)
        print("Generated text (first 500 chars):")
        print(generated_text[:500] + ("..." if len(generated_text) > 500 else ""))


def loop(config_path: Path) -> None:
    """Run the complete pipeline: prepare -> train -> sample."""
    print(f"[gemma_finetuning_mps] Running complete pipeline from: {config_path}")
    prepare_from_toml(config_path)
    train_from_toml(config_path)
    sample_from_toml(config_path)
    print("[gemma_finetuning_mps] Pipeline completed successfully!")