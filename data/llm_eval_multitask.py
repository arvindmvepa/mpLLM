"""
Medical‑VQA multi‑task grader (logit version, +unknown_logis support)

Run:
    python grade_vqa_logits.py --gt gt_file.json --pred pred_file.json
"""

import argparse, json, math
from collections import Counter
from pathlib import Path
from typing import List, Union

# ----------------------------------------------------------------------
# helper maps (index‑0 = “not asked” sentinel everywhere)
# ----------------------------------------------------------------------
AREA_LABS = ["not‑asked", "n/a", "<1%", "1‑5%", "5‑10%",
              "10‑25%", "25‑50%", "50‑75%"]
SHAPE_LABS = ["not‑asked", "n/a", "focus", "round", "oval",
              "elongated", "irregular"]
SAT_LABS = ["not‑asked", "n/a", "single lesion",
              "core with satellite lesions", "scattered lesions"]
LOBES = ["not‑asked", "n/a", "frontal", "parietal", "occipital", "temporal",
         "limbic", "insula", "subcortical", "cerebellum", "brainstem"]

def load_json(path: Union[str, Path]) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def argmax_idx(arr: List[float]) -> int:
    return max(range(len(arr)), key=arr.__getitem__)

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def evaluate_pair(gt_d, pred_d, correct, total):
    gt = gt_d["answer_vqa_numeric"]

    # ---------- Task‑1: AREA (single‑label) ----------
    if gt[0] != 0:
        total["area"] += 1
        pred_lab = argmax_idx(pred_d["area_logits"])
        if pred_lab != 0 and pred_lab == gt[0]:
            correct["area"] += 1

    # ---------- Task‑2: REGION (multi‑label) ----------
    if gt[1] not in (0, [0]):
        total["region"] += 1
        logits = pred_d["region_logits"]
        pred_lobes = {i for i, l in enumerate(logits[1:], start=1)
                      if sigmoid(l) >= 0.5}
        gt_lobes = set(gt[1]) if isinstance(gt[1], list) else {gt[1]}

        intersection = len(pred_lobes.intersection(gt_lobes))
        union = len(pred_lobes.union(gt_lobes))

        tp = intersection
        total_lobes = len(logits)
        tn = total_lobes - union
        acc = (tp+tn)/total_lobes

        correct["region"] += acc

        """
        #IOU calculation
        if union == 0 and intersection == 0:
            correct["region"] += 1
        elif union == 0 or intersection == 0:
            correct["region"] += 0
        else:
            correct["region"] += (intersection/union)
        """

    # ---------- Task‑3: SHAPE (single‑label) ----------
    if gt[2] != 0:
        total["shape"] += 1
        pred_lab = argmax_idx(pred_d["shape_logits"])
        if pred_lab != 0 and pred_lab == gt[2]:
            correct["shape"] += 1

    # ---------- Task‑4: SATELLITE (single‑label) ----------
    if gt[3] != 0:
        total["satellite"] += 1
        pred_lab = argmax_idx(pred_d["satellite_logits"])
        if pred_lab != 0 and pred_lab == gt[3]:
            correct["satellite"] += 1

    # ---------- Task‑5: UNKNOWN ----------
    # Only score when GT says the question *was* about "unknown" (gt == 1)
    if gt[4] != 0:
        total["unknown"] += 1

        pred_unknown = sigmoid(pred_d["unknown_logits"][0]) > 0

        if pred_unknown:
            correct["unknown"] += 1


# ----------------------------------------------------------------------
# main driver
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file",   required=True, help="Ground‑truth JSON file")
    parser.add_argument("--pred_file", required=True, help="Predictions JSON file")
    parser.add_argument("--out_file", required=True, help="Where to write aggregated JSON")
    args = parser.parse_args()

    gt, pred = load_json(args.gt_file), load_json(args.pred_file)
    assert len(gt) == len(pred), "GT and prediction files differ in length!"

    correct, total = Counter(), Counter()
    for g, pr in zip(gt, pred):
        evaluate_pair(g, pr, correct, total)

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------
    print("\nPer‑task accuracies")
    print("-------------------")
    results = dict()
    for task in ["area", "region", "shape", "satellite", "unknown"]:
        if total[task]:
            acc = correct[task] / total[task]
            results[task] = acc
            print(f"{task:10s}: {acc:5.3f}  ({correct[task]}/{total[task]})")
        else:
            print(f"{task:10s}: n/a   (0/0)")
            results[task] = 0.0
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
