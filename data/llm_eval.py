#!/usr/bin/env python3
"""
Medical-VQA multi-task evaluator
================================

Each subset now contains **15 model-related numbers**:

| core | majority-v1 | always-zero | non-zero |
|------|-------------|------------|----------|
| area_acc | area_maj_acc | area_zero_acc | area_nz_acc |
| region_iou | region_maj_iou | region_zero_iou | region_nz_iou |
| shape_acc | shape_maj_acc | shape_zero_acc | shape_nz_acc |
| satellite_acc | satellite_maj_acc | satellite_zero_acc | satellite_nz_acc |
| unknown_acc | unknown_maj_acc | unknown_zero_acc | unknown_nz_acc |

“Non-zero” means
* area/shape/satellite  → GT ≠ 0 **and** Pred ≠ 0
* unknown/partially_unknown  → GT == 1 **and** Pred == 1
* region  → both masks have ≥ 1 cube

Outputs:
    overall → question_type_scores → label_scores
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict, OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


# ----------------------------------------------------------------------
# Region-mask IoU helper (0/0 → 1.0)
# ----------------------------------------------------------------------
def iou_mask(a: Sequence[int], b: Sequence[int]) -> float:
    ta, tb = torch.tensor(a).bool(), torch.tensor(b).bool()
    if not ta.any() and not tb.any():
        return 1.0
    if not ta.any() or not tb.any():
        return 0.0
    inter = (ta & tb).sum().item()
    union = (ta | tb).sum().item()
    return inter / union if union else 0.0


# ----------------------------------------------------------------------
class VQAEval:
    TASKS = ("area", "region", "shape", "satellite", "unknown")

    CORE_KEYS = (
        "area_acc",
        "region_iou",
        "shape_acc",
        "satellite_acc",
        "unknown_acc",
    )
    MAJ_KEYS = (
        "area_maj_acc",
        "region_maj_iou",
        "shape_maj_acc",
        "satellite_maj_acc",
        "unknown_maj_acc",
    )
    ZER_KEYS = (
        "area_zero_acc",
        "region_zero_iou",
        "shape_zero_acc",
        "satellite_zero_acc",
        "unknown_zero_acc",
    )
    NZ_KEYS = (
        "area_nz_acc",
        "region_nz_iou",
        "shape_nz_acc",
        "satellite_nz_acc",
        "unknown_nz_acc",
    )
    OVER_KEY = ("overall_acc",)

    def __init__(self, gt_path: str, pred_path: str, *args, **kwargs) -> None:
        if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
            raise FileNotFoundError("GT or prediction JSON missing")

        self.gt = self._load(gt_path)
        self.pr = self._load(pred_path)
        if len(self.gt) != len(self.pr):
            raise ValueError("GT and prediction list lengths differ")

        # label → qtype → list[(targets_tuple, preds_tuple, mt_tuple)]
        self._data: Dict[
            str,
            Dict[str, List[Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[float, ...]]]],
        ] = defaultdict(lambda: defaultdict(list))

        self.final: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def _load(path: str) -> List[Mapping[str, Any]]:
        with open(path, "r") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def evaluate(self) -> None:
        self._collect()
        self._score()

    # ------------------------------------------------------------------
    # 1 · Collect all multi-task items
    # ------------------------------------------------------------------
    def _collect(self) -> None:
        for g, p in zip(self.gt, self.pr):
            label = g.get("label_name", "NoLabel")
            qtype = g.get("content_type", "unknown")

            if qtype not in {
                "area",
                "shape",
                "satellite",
                "unknown",
                "partially_unknown",
                "region",
            }:
                continue  # skip non-multi-task

            # ---------- ground-truth targets ----------
            gt_area = int(torch.tensor(g.get("area_label", p["area_label"])).argmax())
            gt_shape = int(torch.tensor(g.get("shape_label", p["shape_label"])).argmax())
            gt_sat = int(torch.tensor(g.get("satellite_label", p["satellite_label"])).argmax())
            gt_unk = int(g.get("unknown_label", p.get("unknown_label", [0]))[0])
            gt_reg = tuple(g.get("region_label", p["region_label"]))

            # ---------- model predictions --------------
            pr_area = int(torch.tensor(p["area_logits"]).argmax())
            pr_shape = int(torch.tensor(p["shape_logits"]).argmax())
            pr_sat = int(torch.tensor(p["satellite_logits"]).argmax())
            pr_unk = int(p.get("unknown_logits", [0.0])[0] > 0)
            pr_reg = tuple(
                (torch.tensor(p["region_logits"]).sigmoid() > 0.5).int().tolist()
            )

            targets = (gt_area, gt_reg, gt_shape, gt_sat, gt_unk)
            preds = (pr_area, pr_reg, pr_shape, pr_sat, pr_unk)
            mt_tuple = self._model_metrics(targets, preds)

            self._data[label][qtype].append((targets, preds, mt_tuple))

    # ..........................................................
    def _model_metrics(
        self, tgt: Tuple[Any, ...], prd: Tuple[Any, ...]
    ) -> Tuple[float, ...]:
        gt_area, gt_reg, gt_shape, gt_sat, gt_unk = tgt
        pr_area, pr_reg, pr_shape, pr_sat, pr_unk = prd

        area_acc = float(pr_area == gt_area)
        shape_acc = float(pr_shape == gt_shape)
        sat_acc = float(pr_sat == gt_sat)
        unk_acc = float(pr_unk == gt_unk)

        if not any(gt_reg) and not any(pr_reg):
            reg_iou = 1.0
        elif not any(gt_reg) or not any(pr_reg):
            reg_iou = 0.0
        else:
            reg_iou = iou_mask(gt_reg, pr_reg)

        return (area_acc, reg_iou, shape_acc, sat_acc, unk_acc)

    # ------------------------------------------------------------------
    # 2 · Metrics for one subset
    # ------------------------------------------------------------------
    def _subset_stats(
        self,
        items: List[
            Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[float, ...]]
        ],
    ) -> Dict[str, Any]:
        tgts, prds, mts = zip(*items)
        n = len(items)
        stats: Dict[str, Any] = {"count": n}

        # ---- core model metrics ----
        mt_arr = np.asarray(mts, float)
        for idx, key in enumerate(self.CORE_KEYS):
            stats[key] = float(mt_arr[:, idx].mean()) if n else 0.0

        stats["overall_acc"] = (sum(stats[k] for k in self.CORE_KEYS) /
                                len(self.CORE_KEYS)) if n else 0.0  # ★

        # split targets/preds by task
        tgt_by = list(zip(*tgts))  # five lists
        prd_by = list(zip(*prds))

        for idx, task in enumerate(self.TASKS):
            # ----- majority baseline (local mode) -----
            mode_val = Counter(tgt_by[idx]).most_common(1)[0][0]
            if task == "region":
                maj = [iou_mask(t, mode_val) for t in tgt_by[idx]]
                zero = [iou_mask(t, (0,) * 9) for t in tgt_by[idx]]
            else:
                maj = [float(t == mode_val) for t in tgt_by[idx]]
                zero = [float(t == 0) for t in tgt_by[idx]]
            stats[self.MAJ_KEYS[idx]] = float(np.mean(maj))
            stats[self.ZER_KEYS[idx]] = float(np.mean(zero))

            # ----- non-zero metric -----
            if task == "region":
                mask = [any(gt) and any(pr) for gt, pr in zip(tgt_by[idx], prd_by[idx])]
                nz_scores = [
                    iou_mask(gt, pr)
                    for gt, pr, m in zip(tgt_by[idx], prd_by[idx], mask)
                    if m
                ]
            elif task == "unknown":
                mask = [
                    gt == 1 and pr == 1 for gt, pr in zip(tgt_by[idx], prd_by[idx])
                ]
                nz_scores = [1.0] * sum(mask)  # correct whenever mask true
            else:
                mask = [
                    gt != 0 and pr != 0 for gt, pr in zip(tgt_by[idx], prd_by[idx])
                ]
                nz_scores = [
                    float(gt == pr)
                    for gt, pr, m in zip(tgt_by[idx], prd_by[idx], mask)
                    if m
                ]

            stats[self.NZ_KEYS[idx]] = (
                float(np.mean(nz_scores)) if nz_scores else 0.0
            )

        return stats

    # ..........................................................
    def _aggregate(self, sets: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        sets = list(sets)
        total = sum(s["count"] for s in sets)
        out: Dict[str, Any] = {"count": total}
        for k in (
            *self.CORE_KEYS,
            *self.MAJ_KEYS,
            *self.ZER_KEYS,
            *self.NZ_KEYS,
            *self.OVER_KEY,
        ):
            out[k] = 0.0
        if not total:
            return out
        for k in (
            *self.CORE_KEYS,
            *self.MAJ_KEYS,
            *self.ZER_KEYS,
            *self.NZ_KEYS,
            *self.OVER_KEY,
        ):
            out[k] = round(sum(s[k] * s["count"] for s in sets) / total, 4)
        return out

    # ------------------------------------------------------------------
    # 3 · Aggregate to label / question-type / overall
    # ------------------------------------------------------------------
    def _score(self) -> None:
        label_scores: Dict[str, Dict[str, Any]] = {}
        qtype_sets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        global_sets: List[Dict[str, Any]] = []

        for lbl, qdict in self._data.items():
            label_scores[lbl] = {}
            per_lbl = []
            for qtype, items in qdict.items():
                sub = self._subset_stats(items)
                label_scores[lbl][qtype] = sub
                per_lbl.append(sub)
                qtype_sets[qtype].append(sub)
                global_sets.append(sub)
            label_scores[lbl]["overall"] = self._aggregate(per_lbl)

        # collapse question-type lists
        qtype_scores = {ct: self._aggregate(sets) for ct, sets in qtype_sets.items()}
        qtype_scores["overall"] = self._aggregate(qtype_scores.values())

        self.final = OrderedDict(
            overall=self._aggregate(global_sets),
            question_type_scores=qtype_scores,
            label_scores=label_scores,
        )

    def write_results(self, out_path: str) -> None:
        with open(out_path, "w") as f:
            json.dump(self.final, f, indent=4)
        print(json.dumps(self.final, indent=4))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Evaluate Medical-VQA multi-task metrics + baselines")
    ap.add_argument("--gt_file", required=True, help="Ground-truth JSON list")
    ap.add_argument("--pred_file", required=True, help="Prediction JSON list")
    ap.add_argument("--out_file", required=True, help="Where to write aggregated JSON")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    ev = VQAEval(args.gt_file, args.pred_file)
    ev.evaluate()
    ev.write_results(args.out_file)


if __name__ == "__main__":
    main()
