"""EEG data loader: windows from the per-recording .npz grids, never across recordings or 60 s chunks.

The grids are written by data/<dataset>/prepare.py, one file per recording in data/<dataset>/{train,val}/:
  codes [C, T] uint16, segment_ids [T] (which 60 s chunk a frame came from), channel_names [C], xyz [C, 3].

Single channel now: get_batch -> x, y [B, T], like nanoGPT's text batches (y = x shifted by one token).
Multi-channel later: get_grid_batch -> [B, C, T] plus xyz, from the same index (sketched at the bottom).
"""
import glob
import os

import numpy as np
import torch


class EEGDataLoader:
    def __init__(self, data_dir, split, block_size):
        self.recs, runs = [], []
        for f in sorted(glob.glob(os.path.join(data_dir, split, '*.npz'))):
            d = np.load(f)
            r = len(self.recs)
            loc = d['location_id'].astype(np.int64) if 'location_id' in d.files else np.full(len(d['codes']), -1)
            self.recs.append(dict(codes=d['codes'], xyz=d['xyz'], names=d['channel_names'],
                                  loc=loc, std_rows=np.flatnonzero(loc >= 0)))
            seg = d['segment_ids']
            bounds = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]      # 60 s chunk edges, in frames
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                n = hi - lo - block_size                                        # valid start times (x + one target token)
                if n > 0:
                    runs.append((r, lo, n, d['codes'].shape[0]))
        assert runs, f"no chunk in {data_dir=}/{split=} is longer than block_size={block_size} frames"
        self.runs = np.array(runs, dtype=np.int64)                              # (recording, first start, n starts, n channels)
        w = self.runs[:, 2] * self.runs[:, 3]                                   # every (channel, time) window equally likely
        self.p_single = w / w.sum()
        has_std = np.array([len(self.recs[r]['std_rows']) > 0 for r in self.runs[:, 0]])
        self.p_grid = self.runs[:, 2] * has_std                                 # every start time equally likely
        self.p_grid = self.p_grid / self.p_grid.sum()                           # (recordings with no usable channel skipped)
        self.block_size = block_size
        print(f"{split}: {len(self.recs)} recordings, {len(self.runs)} chunks, {int(w.sum()):,} single-channel windows")

    def get_batch(self, batch_size):
        """Single channel: x, y [B, T] int64, y is x shifted by one token."""
        k = np.random.choice(len(self.runs), size=batch_size, p=self.p_single)
        x, y = [], []
        for r, lo, n, n_ch in self.runs[k]:
            s = lo + np.random.randint(n)                                       # start time inside the chunk
            ch = np.random.randint(n_ch)                                        # one channel of that recording
            w = self.recs[r]['codes'][ch, s:s + self.block_size + 1].astype(np.int64)
            x.append(w[:-1])
            y.append(w[1:])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))

    # Multi-channel
    def get_grid_batch(self, batch_size, max_channels, channel_select='standard94', extra_frames=0):
        """Multi-channel windows. x [B, C, T] ids (T = the length given to __init__, C = max_channels),
        and a dict: loc [B, C] standard location id (-1 = none), xyz [B, C, 3], valid [B, C] (False = padded row).
        channel_select: 'standard94' = one channel per standard location (location_id >= 0); 'all' = every channel.
        extra_frames: 1 for next-token prediction (x gets T + 1 frames; the caller splits it into input and target).
        Recordings with more usable channels than max_channels contribute a random subset."""
        assert extra_frames in (0, 1), "the index only guarantees one frame past the window"
        T = self.block_size + extra_frames
        k = np.random.choice(len(self.runs), size=batch_size, p=self.p_grid)
        x = np.zeros((batch_size, max_channels, T), np.int64)
        loc = np.full((batch_size, max_channels), -1, np.int64)
        xyz = np.zeros((batch_size, max_channels, 3), np.float32)
        valid = np.zeros((batch_size, max_channels), bool)
        for b, (r, lo, n, _) in enumerate(self.runs[k]):
            rec = self.recs[r]
            rows = rec['std_rows'] if channel_select == 'standard94' else np.arange(len(rec['codes']))
            rows = np.random.permutation(rows)[:max_channels]                   # random order, at most max_channels
            s = lo + np.random.randint(n)                                       # one start time for all channels
            c = len(rows)
            x[b, :c] = rec['codes'][rows, s:s + T]
            loc[b, :c] = rec['loc'][rows]
            xyz[b, :c] = rec['xyz'][rows]
            valid[b, :c] = True
        return torch.from_numpy(x), dict(loc=torch.from_numpy(loc), xyz=torch.from_numpy(xyz), valid=torch.from_numpy(valid))

class EEGLabelledLoader:   #jm
    """Labelled windows (task='classify'). data/<dataset>/windows.npz says where every window starts and what its
    label is; the token grids are data/<dataset>/{train,val,test}/*.npz (both from data/eeg_labelled/prepare.py)."""
    SPLITS = {'train': 0, 'val': 1, 'test': 2}

    def __init__(self, data_dir, split):
        self.recs = {}
        for f in sorted(glob.glob(os.path.join(data_dir, split, '*.npz'))):
            d = np.load(f)
            loc = d['location_id'].astype(np.int64)
            self.recs[int(d['recording_id'])] = dict(codes=d['codes'], xyz=d['xyz'], loc=loc, std_rows=np.flatnonzero(loc >= 0))
        w = np.load(os.path.join(data_dir, 'windows.npz'))
        keep = (w['split'] == self.SPLITS[split]) & np.isin(w['recording_id'], list(self.recs))
        self.rec_id, self.start, self.label = w['recording_id'][keep], w['start_frame'][keep], w['label'][keep]
        self.n_frames = int(w['n_frames'][keep][0])
        counts = np.bincount(self.label)
        self.p_balanced = 1.0 / counts[self.label]                              # rare classes drawn more often
        self.p_balanced /= self.p_balanced.sum()
        print(f"{split}: {len(self.recs)} recordings, {len(self.label)} windows of {self.n_frames} frames, per class {counts.tolist()}")

    def _grid(self, idx, max_channels, channel_select, shuffle_channels):
        """Windows idx -> x [B, C, T], dict(loc, xyz, valid) as in get_grid_batch, y [B] class ids."""
        B, T = len(idx), self.n_frames
        x = np.zeros((B, max_channels, T), np.int64)
        loc = np.full((B, max_channels), -1, np.int64)
        xyz = np.zeros((B, max_channels, 3), np.float32)
        valid = np.zeros((B, max_channels), bool)
        for b, i in enumerate(idx):
            rec, s = self.recs[self.rec_id[i]], self.start[i]
            rows = rec['std_rows'] if channel_select == 'standard94' else np.arange(len(rec['codes']))
            rows = (np.random.permutation(rows) if shuffle_channels else rows)[:max_channels]
            c = len(rows)
            x[b, :c] = rec['codes'][rows, s:s + T]
            loc[b, :c] = rec['loc'][rows]
            xyz[b, :c] = rec['xyz'][rows]
            valid[b, :c] = True
        kw = dict(loc=torch.from_numpy(loc), xyz=torch.from_numpy(xyz), valid=torch.from_numpy(valid))
        return torch.from_numpy(x), kw, torch.from_numpy(self.label[idx])

    def get_batch(self, batch_size, max_channels, channel_select='standard94'):
        """Random training windows, every class equally likely."""
        idx = np.random.choice(len(self.label), size=batch_size, p=self.p_balanced)
        return self._grid(idx, max_channels, channel_select, shuffle_channels=True)

    def iter_all(self, batch_size, max_channels, channel_select='standard94'):
        """Every window once, in order, fixed channel subset: for evaluation."""
        for i in range(0, len(self.label), batch_size):
            yield self._grid(np.arange(i, min(i + batch_size, len(self.label))), max_channels, channel_select, shuffle_channels=False)
