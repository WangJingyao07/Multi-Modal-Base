#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, re
from glob import glob

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit



EMO_MAP = {"ANG":"ANG", "DIS":"DIS", "FEA":"FEA", "HAP":"HAP", "NEU":"NEU", "SAD":"SAD"}



def parse_cremad(path):
    """Return (filepath, label, subject_id) parsed from CREMA-D filename."""
    fn = os.path.basename(path)
    # robust split: allow extra underscores in middle tokens, we only need 0 and 2
    toks = fn.split("_")
    if len(toks) < 4 or not toks[0].isdigit():
        raise ValueError(f"Unexpected CREMA-D filename: {fn}")
    
    subject_id = toks[0]
    emo = toks[2].upper()
    if emo not in EMO_MAP:
        raise ValueError(f"Unknown emotion '{emo}' in {fn}")
    return path, EMO_MAP[emo], subject_id



def save_list(items, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for it in items:
            f.write(str(it).strip() + "\n")

# def save_list(items, outpath):
#     os.makedirs(os.path.dirname(outpath), exist_ok=True)
#     with open(outpath, "w", encoding="utf-8") as f:
#         for it in items:
#             f.write(str(it).strip() + "\n")



def subject_disjoint_split(df, test_size=0.2, val_size=0.1, seed=42):
    X = df["filepath"].values

    y = df["label"].values
    g = df["subject_id"].values


    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(outer.split(X, y, groups=g))
    df_train = df.iloc[tr_idx].reset_index(drop=True)
    df_test  = df.iloc[te_idx].reset_index(drop=True)

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr2_idx, va_idx = next(inner.split(df_train["filepath"].values,
                                       df_train["label"].values,
                                       groups=df_train["subject_id"].values))
    df_tr = df_train.iloc[tr2_idx].reset_index(drop=True)
    df_va = df_train.iloc[va_idx].reset_index(drop=True)
    # df_va = df_train.iloc[va_idx].reset_index(drop=True)




    s_tr, s_va, s_te = set(df_tr["subject_id"]), set(df_va["subject_id"]), set(df_test["subject_id"])
    assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te), "Subject leakage detected!"
    return df_tr, df_va, df_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cremad_root", required=True, help="Path to CREMA-D root (folder containing WAVs)")
    ap.add_argument("--outdir", default="dataset/CREMA-D/splits", help="Where to save split lists")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()


    wavs = glob(os.path.join(args.cremad_root, "**", "*.wav"), recursive=True)
    if not wavs:
        raise FileNotFoundError(f"No WAV files found under: {args.cremad_root}")

    rows = []
    for w in sorted(wavs):
        try:
            fp, lab, sid = parse_cremad(w)
            rows.append({"filepath": fp, "label": lab, "subject_id": sid})
        except Exception as e:
            # You can choose to skip or raise; we raise to keep dataset clean.
            raise

    df = pd.DataFrame(rows).drop_duplicates()
    df_tr, df_va, df_te = subject_disjoint_split(df, test_size=args.test_size, val_size=args.val_size, seed=args.seed)

    # Save file lists
    save_list(df_tr["filepath"], os.path.join(args.outdir, "train_files.txt"))
    save_list(df_va["filepath"], os.path.join(args.outdir, "val_files.txt"))
    save_list(df_te["filepath"], os.path.join(args.outdir, "test_files.txt"))

    summary = {
        "counts": {"train": len(df_tr), "val": len(df_va), "test": len(df_te), "all": len(df)},
        "subjects": {
            "train": len(set(df_tr["subject_id"])),
            "val":   len(set(df_va["subject_id"])),
            "test":  len(set(df_te["subject_id"])),
            "all":   len(set(df["subject_id"]))
        },
        "test_size": args.test_size,
        "val_size": args.val_size,
        "seed": args.seed
    }
    
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Subject-disjoint splits created at:", args.outdir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
