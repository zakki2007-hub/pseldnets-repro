"""Evaluate SELD predictions against synthetic-test-set ground truth.

Works around the class-index bug in src/evaluate.py (it prints per-class rows
for nb_classes=170 while only 158 classes are present -> IndexError) and adds
per-room / per-polyphony breakdowns.

Usage:
    python _analysis/eval_analysis.py --pred_dir <dir with *.csv> \
        --gt_dir datasets/test360_ov3/metadata \
        --label_h5 _hdf5/label/accdoa/dev/test360_ov3.h5 \
        --out_dir _analysis
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.data_utilities import load_output_format_file, to_metrics_format  # noqa: E402
from utils.SELD_metrics import SELDMetrics  # noqa: E402

EPS = np.finfo(np.float32).eps
FIELDS = ["TP", "FP", "FP_spatial", "FN", "Nref", "total_DE", "DE_TP", "DE_FN"]


def snapshot(ev):
    d = {f: getattr(ev, "_" + f).copy() for f in FIELDS}
    d["S"], d["D"], d["I"] = ev._S, ev._D, ev._I
    return d


def diff(a, b):
    d = {f: a[f] - b[f] for f in FIELDS}
    d["S"], d["D"], d["I"] = a["S"] - b["S"], a["D"] - b["D"], a["I"] - b["I"]
    return d


def aggregate(per_file, files):
    agg = {f: np.zeros_like(per_file[files[0]][f]) for f in FIELDS}
    agg["S"] = agg["D"] = agg["I"] = 0
    for f in files:
        for k in agg:
            agg[k] = agg[k] + per_file[f][k]
    return agg


def summarize(agg):
    TP, FP, FPs, FN = agg["TP"], agg["FP"], agg["FP_spatial"], agg["FN"]
    Nref, DE, DE_TP, DE_FN = agg["Nref"], agg["total_DE"], agg["DE_TP"], agg["DE_FN"]
    ER = (agg["S"] + agg["D"] + agg["I"]) / (Nref.sum() + EPS)

    F = TP / (EPS + TP + FPs + 0.5 * (FP + FN))
    LE = DE / (DE_TP + EPS)
    LE[DE_TP == 0] = 180.0
    LR = DE_TP / (EPS + DE_TP + DE_FN)
    cls = np.mean([np.repeat(ER, len(TP)), 1 - F, LE / 180.0, 1 - LR], axis=0)
    present = DE_TP > 0
    if not present.any():
        return {
            "macro": {"ER": float(ER), "F": 0.0, "LE": 180.0, "LR": 0.0,
                      "SELD_scr": 1.0, "n_classes_present": 0},
            "micro": {"ER": float(ER), "F": 0.0, "LE": 180.0, "LR": 0.0,
                      "SELD_scr": 1.0},
            "sed_counts": {"S": int(agg["S"]), "D": int(agg["D"]), "I": int(agg["I"]),
                           "Nref": int(Nref.sum())},
            "detection": {"TP": 0, "FP": int(FP.sum()), "FP_spatial": int(FPs.sum()),
                          "FN": int(FN.sum()), "recall_at_20deg": 0.0, "precision_at_20deg": 0.0},
            "classwise": {"present_idx": [], "ER": float(ER), "F": [], "LE": [], "LR": [],
                          "SELD_scr": [], "nref": []},
        }

    F_micro = TP.sum() / (EPS + TP.sum() + FPs.sum() + 0.5 * (FP.sum() + FN.sum()))
    LE_micro = DE.sum() / (DE_TP.sum() + EPS) if DE_TP.sum() else 180.0
    LR_micro = DE_TP.sum() / (EPS + DE_TP.sum() + DE_FN.sum())
    seld_micro = float(np.mean([ER, 1 - F_micro, LE_micro / 180.0, 1 - LR_micro]))

    return {
        "macro": {
            "ER": float(ER),
            "F": float(F[present].mean()),
            "LE": float(LE[present].mean()),
            "LR": float(LR[present].mean()),
            "SELD_scr": float(cls[present].mean()),
            "n_classes_present": int(present.sum()),
        },
        "micro": {
            "ER": float(ER),
            "F": float(F_micro),
            "LE": float(LE_micro),
            "LR": float(LR_micro),
            "SELD_scr": seld_micro,
        },
        "sed_counts": {"S": int(agg["S"]), "D": int(agg["D"]), "I": int(agg["I"]),
                       "Nref": int(Nref.sum())},
        "detection": {
            "TP": int(TP.sum()), "FP": int(FP.sum()), "FP_spatial": int(FPs.sum()),
            "FN": int(FN.sum()),
            "recall_at_20deg": float(TP.sum() / (TP.sum() + FN.sum() + EPS)),
            "precision_at_20deg": float(TP.sum() / (TP.sum() + FP.sum() + FPs.sum() + EPS)),
        },
        "classwise": {
            "present_idx": np.where(present)[0].tolist(),
            "ER": float(ER),
            "F": F[present].tolist(),
            "LE": LE[present].tolist(),
            "LR": LR[present].tolist(),
            "SELD_scr": cls[present].tolist(),
            "nref": Nref[present].astype(int).tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", default="datasets/test360_ov3/metadata")
    ap.add_argument("--label_h5", default="_hdf5/label/accdoa/dev/test360_ov3.h5")
    ap.add_argument("--nb_classes", type=int, default=170)
    ap.add_argument("--out_dir", default="_analysis")
    args = ap.parse_args()

    gt_files = sorted(Path(args.gt_dir).glob("*.csv"))
    pred_files = {p.stem: p for p in Path(args.pred_dir).glob("*.csv")}
    missing = [g.stem for g in gt_files if g.stem not in pred_files]
    # evaluate only on clips that have predictions (allows room-wise splits)
    gt_files = [g for g in gt_files if g.stem in pred_files]
    extra = [s for s in pred_files if s not in {g.stem for g in gt_files}]

    # pre-load GT labels once
    gt_labels, gt_num_frames = {}, {}
    for g in gt_files:
        d = load_output_format_file(g)
        gt_num_frames[g.stem] = max(d.keys())
        gt_labels[g.stem] = to_metrics_format(d, max(d.keys()), label_resolution=0.1)

    ev = SELDMetrics(nb_classes=args.nb_classes, doa_threshold=20)
    per_file = {}
    prev = snapshot(ev)
    for g in gt_files:
        stem = g.stem
        if pred_files[stem].stat().st_size == 0:
            pred = {}
        else:
            pred = load_output_format_file(pred_files[stem])
        pred_labels = to_metrics_format(pred, gt_num_frames[stem], label_resolution=0.1)
        ev.update_seld_scores(pred_labels, gt_labels[stem])
        cur = snapshot(ev)
        per_file[stem] = diff(cur, prev)
        prev = cur

    all_files = [g.stem for g in gt_files]
    overall = summarize(aggregate(per_file, all_files))

    # per room (fold1_roomX)
    rooms = {}
    for stem in all_files:
        rooms.setdefault(stem.split("_mix")[0], []).append(stem)
    per_room = {r: summarize(aggregate(per_file, fs)) for r, fs in sorted(rooms.items())}

    # per max polyphony from GT labels (170-class se matrix)
    poly = {}
    with h5py.File(args.label_h5, "r") as h5:
        for stem in all_files:
            se = h5[stem]["accdoa"]["se"][:]
            max_poly = int(max(1, se.sum(axis=1).max()))
            poly.setdefault(max_poly, []).append(stem)
    per_poly = {str(k): summarize(aggregate(per_file, fs)) for k, fs in sorted(poly.items())}

    # per-file distribution (micro over one clip)
    file_scores = {stem: summarize(per_file[stem])["micro"]["SELD_scr"] for stem in all_files}
    vals = np.array(list(file_scores.values()))
    worst = sorted(file_scores.items(), key=lambda kv: kv[1])[:5]

    out = {
        "pred_dir": str(Path(args.pred_dir).resolve()),
        "n_gt_files": len(gt_files),
        "n_pred_files": len(pred_files),
        "missing_preds": missing,
        "extra_preds": extra,
        "overall": overall,
        "per_room": per_room,
        "per_max_polyphony": per_poly,
        "per_file_micro_seld": {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p10": float(np.quantile(vals, 0.10)),
            "p90": float(np.quantile(vals, 0.90)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "worst_files": worst,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    # classwise csv
    cw = overall["classwise"]
    with (out_dir / "classwise.csv").open("w") as f:
        f.write("class_id,ER,F,LE,LR,SELD_scr,nref_frames\n")
        for i, cid in enumerate(cw["present_idx"]):
            f.write(f"{cid},{cw['ER']:.6f},{cw['F'][i]:.6f},{cw['LE'][i]:.6f},"
                    f"{cw['LR'][i]:.6f},{cw['SELD_scr'][i]:.6f},{cw['nref'][i]}\n")

    def fmt(m):
        return (f"  ER {m['ER']:.4f} | F {100*m['F']:5.2f}% | LE {m['LE']:6.2f}° | "
                f"LR {100*m['LR']:5.2f}% | SELD {m['SELD_scr']:.4f}")

    print(f"files: gt={len(gt_files)} pred={len(pred_files)} missing={len(missing)} extra={len(extra)}")
    print(f"present classes: {overall['macro']['n_classes_present']}/{args.nb_classes}")
    print("\n[ALL]")
    print("  macro:" + fmt(overall["macro"]))
    print("  micro:" + fmt(overall["micro"]))
    sc = overall["sed_counts"]
    print(f"  SED counts: Nref={sc['Nref']} S={sc['S']} D={sc['D']} I={sc['I']}")
    det = overall["detection"]
    print(f"  detection: TP={det['TP']} FP={det['FP']} FP_spatial={det['FP_spatial']} FN={det['FN']} | "
          f"recall@20° {100*det['recall_at_20deg']:.2f}% precision@20° {100*det['precision_at_20deg']:.2f}%")
    print("\n[per room]")
    for r, m in per_room.items():
        print(f"  {r:12s} macro:" + fmt(m["macro"]) + f"  | micro SELD {m['micro']['SELD_scr']:.4f}")
    print("\n[per max polyphony]")
    for r, m in per_poly.items():
        print(f"  poly {r:3s} n={len(poly[int(r)]):3d} macro:" + fmt(m["macro"]) +
              f"  | micro SELD {m['micro']['SELD_scr']:.4f}")

    order = np.argsort(cw["SELD_scr"])
    print(f"\n[per-clip micro SELD] mean {vals.mean():.3f} | median {np.median(vals):.3f} | "
          f"p10 {np.quantile(vals,0.10):.3f} | p90 {np.quantile(vals,0.90):.3f} | "
          f"min {vals.min():.3f} | max {vals.max():.3f}")
    print("  worst clips:", ", ".join(f"{s}({v:.3f})" for s, v in worst))

    print("\n[worst 10 classes by SELD_scr]  (class_id: SELD / F / LE / LR / nref)")
    for i in order[:10]:
        print(f"  {cw['present_idx'][i]}: {cw['SELD_scr'][i]:.3f} / {100*cw['F'][i]:.1f}% / "
              f"{cw['LE'][i]:.1f} / {100*cw['LR'][i]:.1f}% / {cw['nref'][i]}")
    print("\n[best 10 classes by SELD_scr]")
    for i in order[-10:][::-1]:
        print(f"  {cw['present_idx'][i]}: {cw['SELD_scr'][i]:.3f} / {100*cw['F'][i]:.1f}% / "
              f"{cw['LE'][i]:.1f} / {100*cw['LR'][i]:.1f}% / {cw['nref'][i]}")
    print(f"\nwritten: {out_dir/'results.json'}, {out_dir/'classwise.csv'}")


if __name__ == "__main__":
    main()
