"""EEG tokens: one channel x time grid per recording, plus the flat single-channel stream nanoGPT reads.

Each EEG recording is run through a frozen EEG-DAC checkpoint, one channel at a time, which turns
256 Hz samples into one code id per 32 samples (8 tokens/s, 8192 possible codes). Written:
  train/00000.npz, ...   per recording: codes [channels, frames], segment_ids, channel_names, xyz (see encode_recording)
  train.bin / val.bin    the same codes flattened, every channel one stream after the other, exactly like the
                         characters of data/shakespeare_char -> train.py needs no change: --dataset=eeg_single
  meta.pkl               vocab_size (8192), checkpoint, tokens per second
  codebook.npy           [8192, 8], the DAC's code vectors, for the "feed the vector instead of the id" step
A multi-channel loader later reads the .npz grids and ignores the .bin files.

Run in the DAC environment (dac, audiotools, mne):
  ~/venv_dac_audio/bin/python data/eeg_single/prepare.py                 # 48 train + 8 val recordings, seed 0
  ~/venv_dac_audio/bin/python data/eeg_single/prepare.py --n_train 4 --n_val 1 --device cpu   # quick test
"""
import argparse
import glob
import os
import pickle
import random
import sys
import time
from functools import lru_cache

import numpy as np

DAC_REPO = "/data/groups/bci/jonas/workspace/bci_descript-audio-codec"
CKPT = f"{DAC_REPO}/runs/rand10k_baseline/latest/dac/weights.pth"      # Chris's parameters, 20k steps, 8192 codes
V8 = "/data/groups/bci/datasets/processed/v8_sets/10k"
SR, HOP = 256, 32                                                      # tokens are 32 samples = 125 ms
TARGET_STD, CLIP = 0.3, 1.0                                            # the DAC's EEGNormalize: std 0.3, clip +-1
CHUNK_S, MIN_S = 60, 5                                                 # normalise and encode in 60 s chunks


def load_dac(ckpt, device):
    import torch
    sys.path.insert(0, DAC_REPO)
    from dac.model import DAC
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    kw = blob["metadata"]["kwargs"]
    assert kw["n_codebooks"] == 1 and kw["sample_rate"] == SR, kw
    model = DAC(**kw)
    model.load_state_dict(blob["state_dict"])
    codebook = blob["state_dict"]["quantizer.quantizers.0.codebook.weight"].float().numpy()   # [8192, 8]
    return model.to(device).eval(), codebook

@lru_cache(maxsize=1) #jm
def standard_positions(): #jm
    """The 94 positions of the standard 10-20/10-10 system, in the same head coordinate frame as the data.
    (montage.get_positions() would give a different frame, ~5 cm off; set_montage on an info applies the same transform.)"""
    import mne
    m = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(m.ch_names, SR, "eeg")
    info.set_montage(m)
    return list(m.ch_names), np.array([ch["loc"][:3] for ch in info["chs"]], dtype=np.float32)


def snap_to_standard(xyz, max_dist=0.02): #jm
    """Nearest standard position per channel -> location_id [C] (0..93, -1 = dropped) and its distance [C] in metres.
    Dropped: channels more than max_dist (2 cm) from every standard position, and, where several channels of the
    recording land on the same position, all but the closest one. So at most one channel per position, at most 94."""
    _, P = standard_positions()
    d = np.linalg.norm(xyz[:, None, :] - P[None, :, :], axis=-1)       # [C, 94]
    d = np.where(np.isnan(d), np.inf, d)                                # channels without a position never match
    nearest = d.argmin(1)
    dist = d[np.arange(len(nearest)), nearest]
    loc = np.where(dist <= max_dist, nearest, -1)
    for k in np.unique(loc[loc >= 0]):                                  # one channel per position: keep the closest
        rows = np.flatnonzero(loc == k)
        loc[rows[rows != rows[dist[rows].argmin()]]] = -1
    return loc.astype(np.int16), dist.astype(np.float32)

def encode_recording(path, model, device):
    """One recording -> the channel x time token grid plus what identifies the channels:
         codes        [channels, frames] uint16   (each channel encoded as its own signal)
         segment_ids  [frames] int32             which 60 s chunk a frame came from (chunks are normalised separately)
         channel_names [channels] str            e.g. 'Fz', 'EEG FP1-REF' (as named in the file)
         xyz          [channels, 3] float32      electrode position in head coordinates (metres); NaN if the file has none
    """
    import mne
    import torch
    raw = mne.io.read_raw_fif(path, preload=True, verbose="error")
    raw.pick("eeg", verbose="error")                                   # channels flagged bad are dropped by default
    assert raw.info["sfreq"] == SR, raw.info["sfreq"]
    x = raw.get_data().astype(np.float32)                              # [channels, samples]
    xyz = np.array([ch["loc"][:3] for ch in raw.info["chs"]], dtype=np.float32)
    xyz[(xyz == 0).all(1)] = np.nan                                    # mne stores "unknown" as zeros
    pieces, seg_ids = [], []
    for k, start in enumerate(range(0, x.shape[1], CHUNK_S * SR)):
        seg = x[:, start:start + CHUNK_S * SR]
        seg = seg[:, : seg.shape[1] // HOP * HOP]                      # whole tokens only
        if seg.shape[1] < MIN_S * SR:
            break
        seg = (seg - seg.mean(1, keepdims=True)) * (TARGET_STD / seg.std(1, keepdims=True).clip(1e-8))
        seg = np.clip(seg, -CLIP, CLIP)
        with torch.no_grad():
            xt = torch.from_numpy(seg).unsqueeze(1).to(device)         # [channels, 1, samples]
            _, codes, *_ = model.encode(model.preprocess(xt, SR))      # codes [channels, 1, samples // 32]
        pieces.append(codes[:, 0].cpu().numpy().astype(np.uint16))
        seg_ids.append(np.full(pieces[-1].shape[1], k, dtype=np.int32))
    if not pieces:
        return None
    location_id, location_dist = snap_to_standard(xyz)
    return dict(codes=np.concatenate(pieces, axis=1), segment_ids=np.concatenate(seg_ids),
                channel_names=np.array(raw.ch_names), xyz=xyz, source=path,
                location_id=location_id, location_dist=location_dist)


def pick_recordings(args):
    """A seeded random sample of recordings from the v8 10k train / val folders."""
    random.seed(args.seed)
    return {s: random.sample(sorted(glob.glob(f"{V8}/{s}/**/*.fif", recursive=True)), getattr(args, f"n_{s}"))
            for s in ("train", "val")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--n_train", type=int, default=48)
    ap.add_argument("--n_val", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="which recordings get sampled")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    model, codebook = load_dac(args.ckpt, args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    files = pick_recordings(args)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)
        streams, t0 = [], time.time()
        for i, path in enumerate(files[split]):
            try:
                rec = encode_recording(path, model, args.device)
            except Exception as e:
                print(f"skip {path}: {e}")
                continue
            if rec is None:
                continue
            # 1) the grid, one .npz per recording: what a multi-channel loader needs later
            np.savez(os.path.join(args.out_dir, split, f"{i:05d}.npz"), **rec)
            # 2) flattened: every channel is one token stream, appended -> train.bin / val.bin for the vanilla loader
            streams.extend(rec["codes"])
            print(f"  {split} {os.path.basename(path)[:50]:50s} {rec['codes'].shape[0]:3d} ch x {rec['codes'].shape[1]:6d} tokens")
        ids = np.concatenate(streams).astype(np.uint16)
        ids.tofile(os.path.join(args.out_dir, f"{split}.bin"))
        print(f"{split}: {len(files[split])} recordings, {len(streams)} channel streams, {len(ids):,} tokens, {time.time() - t0:.0f} s")
    np.save(os.path.join(args.out_dir, "codebook.npy"), codebook)
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": int(codebook.shape[0]), "ckpt": args.ckpt, "tokens_per_s": SR / HOP,
                     "locations": standard_positions()[0]}, f)
    print(f"meta.pkl: vocab_size {codebook.shape[0]}; codebook.npy {codebook.shape}")


if __name__ == "__main__":
    main()
