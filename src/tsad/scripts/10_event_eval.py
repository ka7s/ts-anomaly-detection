
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    eval_csv = Path("reports") / "pca_pool_v2_eval.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing eval CSV: {eval_csv}")

    df = pd.read_csv(eval_csv)

    required = {
        "file",
        "anomalous_windows",
        "warning_tp",
        "warning_fp",
        "warning_fn",
        "warning_tn",
        "critical_tp",
        "critical_fp",
        "critical_fn",
        "critical_tn",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eval CSV missing required columns: {sorted(missing)}")

    # Ground-truth event: file contains any anomalous windows
    df["has_event_true"] = df["anomalous_windows"] > 0

    # Predicted event: at least one window predicted anomalous in file
    # window predicted anomalous => TP + FP > 0 (any predicted positive)
    df["has_event_pred_warning"] = (df["warning_tp"] + df["warning_fp"]) > 0
    df["has_event_pred_critical"] = (df["critical_tp"] + df["critical_fp"]) > 0

    def event_confusion(true_col: str, pred_col: str) -> dict[str, int]:
        tp = int((df[true_col] & df[pred_col]).sum())
        fp = int((~df[true_col] & df[pred_col]).sum())
        fn = int((df[true_col] & ~df[pred_col]).sum())
        tn = int((~df[true_col] & ~df[pred_col]).sum())
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def prf(cm: dict[str, int]) -> dict[str, float]:
        tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    cm_w = event_confusion("has_event_true", "has_event_pred_warning")
    cm_c = event_confusion("has_event_true", "has_event_pred_critical")

    met_w = prf(cm_w)
    met_c = prf(cm_c)

    # Useful additional rates
    n_files = int(len(df))
    n_event_files = int(df["has_event_true"].sum())
    n_noevent_files = n_files - n_event_files

    # "False event rate" = fraction of no-event files flagged as events
    fer_w = (cm_w["fp"] / n_noevent_files) if n_noevent_files > 0 else 0.0
    fer_c = (cm_c["fp"] / n_noevent_files) if n_noevent_files > 0 else 0.0

    out = {
        "n_files": n_files,
        "n_event_files_true": n_event_files,
        "n_noevent_files_true": n_noevent_files,
        "event_warning": {
            **cm_w,
            **met_w,
            "false_event_rate": float(fer_w),
        },
        "event_critical": {
            **cm_c,
            **met_c,
            "false_event_rate": float(fer_c),
        },
    }

    out_path = Path("reports") / "pca_pool_v2_event_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Saved event-level summary: {out_path}")

    print("[INFO] Event-level (WARNING / p95): "
          f"P={met_w['precision']:.3f} R={met_w['recall']:.3f} F1={met_w['f1']:.3f} "
          f"FER={fer_w:.3f} (TP={cm_w['tp']}, FP={cm_w['fp']}, FN={cm_w['fn']}, TN={cm_w['tn']})")

    print("[INFO] Event-level (CRITICAL / p99): "
          f"P={met_c['precision']:.3f} R={met_c['recall']:.3f} F1={met_c['f1']:.3f} "
          f"FER={fer_c:.3f} (TP={cm_c['tp']}, FP={cm_c['fp']}, FN={cm_c['fn']}, TN={cm_c['tn']})")


if __name__ == "__main__":
    main()
