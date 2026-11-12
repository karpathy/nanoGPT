"""Labelled EEG windows as DAC tokens, for the classifier (task='classify' in train.py).

Input: one cache.json in Jon Huml's schema v2, as written by the NeuralBench label export
(/data/groups/bci/datasets/processed/v8.1_neuralbench_pilot/tasks/<task>/<Study>/cache.json): recordings
(signal_file, split) and windows (recording_id, start_sample, stop_sample, label; split falls back to the recording's).

Every recording goes through the SAME frozen EEG-DAC as the pretraining data (data/eeg_single/prepare.py: one channel
at a time, 60 s chunks normalised to std 0.3, 8 tokens/s), so pretrained checkpoints apply. Written to --out_dir:
  {train,val,test}/<recording_id>.npz   the eeg_multi grid (codes [channels, frames] uint16, segment_ids, channel_names,
                                        xyz, location_id, location_dist, source) + recording_id
  windows.npz                           recording_id, split (0 train / 1 val / 2 test), start_frame, n_frames, label (int)
  meta.pkl                              as eeg_multi (vocab_size, tokens_per_s, locations) + task, n_classes, label_names,
                                        n_frames, cache
  codebook.npy                          [8192, 8] DAC code vectors (the use_codebook input)
A window's frames are [round(start_sample / 32), + round(duration / 32)). Windows reaching past the encoded frames (the
final partial chunk is not encoded) or crossing a 60 s chunk edge are dropped and counted.

Run in the DAC environment:
  ~/venv_dac_audio/bin/python data/eeg_labelled/prepare.py \
      --cache /data/groups/bci/datasets/processed/v8.1_neuralbench_pilot/tasks/p3/Korczowski2019BrainBi2014B/cache.json \
      --out_dir data/eeg_korczowski_p3
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eeg_single"))
from prepare import CKPT, HOP, SR, encode_array, load_dac, read_fif, standard_positions  # noqa: E402

SPLITS = {"train": 0, "val": 1, "test": 2}


def resolve_signal(cache_path, rel):
    """signal_file is relative to the cache.json's folder or to one of its parents (the pilot's is two up)."""
    d = os.path.dirname(os.path.abspath(cache_path))
    while True:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            return p
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError(f"{rel} not found above {cache_path}")
        d = parent


def label_table(cache):
    """n_classes and the class names, from label_names ({"0": ["NonTarget"], ...}) or label_mapping ({"healthy": 0, ...})."""
    names_by_id = cache.get("label_names") or {}
    if names_by_id:
        n = max(int(k) for k in names_by_id) + 1
        names = ["/".join(v) if isinstance(v, list) else str(v) for _, v in sorted(names_by_id.items(), key=lambda kv: int(kv[0]))]
    else:
        mapping = cache["label_mapping"]
        n = max(mapping.values()) + 1
        names = [None] * n
        for k, v in mapping.items():
            names[v] = k
    return n, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache.json (schema v2)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_recordings", type=int, default=0, help="quick test: encode only the first N")
    args = ap.parse_args()

    cache = json.load(open(args.cache))
    assert cache.get("schema_version") == 2, cache.get("schema_version")
    assert not cache.get("multilabel", False), "multilabel tasks (TUAR) are not supported yet"
    n_classes, label_names = label_table(cache)
    recs = {int(r["recording_id"]): r for r in cache["recordings"]}
    if args.max_recordings:
        recs = dict(list(recs.items())[: args.max_recordings])
    print(f"{cache.get('task')} {cache.get('study', '')}: {len(recs)} recordings, {len(cache['windows'])} windows, "
          f"{n_classes} classes {label_names}")

    model, codebook = load_dac(args.ckpt, args.device)
    for s in SPLITS:
        os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)

    # 1) tokens, one .npz per recording, exactly as for the pretraining data
    n_frames_of, seg_of, t0 = {}, {}, time.time()
    for rid, rec in recs.items():
        path = resolve_signal(args.cache, rec["signal_file"])
        try:
            out = encode_array(*read_fif(path), path, model, args.device)
        except Exception as e:
            print(f"skip recording {rid} ({path}): {e}")
            continue
        if out is None:
            print(f"skip recording {rid}: shorter than 5 s")
            continue
        out["recording_id"] = rid
        np.savez(os.path.join(args.out_dir, rec["split"], f"{rid:05d}.npz"), **out)
        n_frames_of[rid], seg_of[rid] = out["codes"].shape[1], out["segment_ids"]
        print(f"  {rec['split']:5s} {rid:4d} {os.path.basename(path)[:45]:45s} {out['codes'].shape[0]:3d} ch x "
              f"{out['codes'].shape[1]:6d} frames  ({int((out['location_id'] >= 0).sum())} on standard positions)")
    print(f"encoded {len(n_frames_of)} recordings in {time.time() - t0:.0f} s")

    # 2) the window table, in frames
    rows, dropped, lengths = [], Counter(), Counter()
    for w in cache["windows"]:
        rid = int(w["recording_id"])
        if rid not in n_frames_of:
            dropped["recording missing"] += 1
            continue
        start = int(round(w["start_sample"] / HOP))
        n = int(round((w["stop_sample"] - w["start_sample"]) / HOP))
        if start < 0 or start + n > n_frames_of[rid]:
            dropped["outside encoded frames"] += 1
            continue
        if seg_of[rid][start] != seg_of[rid][start + n - 1]:
            dropped["crosses 60 s chunk edge"] += 1
            continue
        lengths[n] += 1
        rows.append((rid, SPLITS[w.get("split", recs[rid]["split"])], start, n, int(w["label"])))
    assert rows, "no window survived"
    n_frames = lengths.most_common(1)[0][0]
    rows = np.array([r for r in rows if r[3] == n_frames], dtype=np.int64)       # one window length per dataset
    np.savez(os.path.join(args.out_dir, "windows.npz"), recording_id=rows[:, 0], split=rows[:, 1].astype(np.int8),
             start_frame=rows[:, 2], n_frames=rows[:, 3], label=rows[:, 4])
    print(f"windows: {len(rows)} kept ({n_frames} frames each), dropped {dict(dropped)}, lengths seen {dict(lengths)}")
    for s, k in SPLITS.items():
        m = rows[:, 1] == k
        print(f"  {s:5s}: {int(m.sum()):7d} windows, per class {np.bincount(rows[m, 4], minlength=n_classes).tolist()}")

    # 3) what train.py reads
    np.save(os.path.join(args.out_dir, "codebook.npy"), codebook)
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": int(codebook.shape[0]), "ckpt": args.ckpt, "tokens_per_s": SR / HOP,
                     "locations": standard_positions()[0],
                     "task": cache.get("task"), "study": cache.get("study"), "cache": os.path.abspath(args.cache),
                     "n_classes": n_classes, "label_names": label_names, "n_frames": n_frames}, f)
    print(f"meta.pkl: vocab_size {codebook.shape[0]}, n_classes {n_classes}, n_frames {n_frames}")


if __name__ == "__main__":
    main()
