import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Union
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None


def add_spaces(text):
    text = " " + text + " "
    text = text.replace(",", " , ")
    text = text.replace(".", " . ")
    text = text.replace(";", " ; ")
    return text


AREA_LABS = ["N/A", "<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-75%"]
AREA_LABS = [add_spaces(lab) for lab in AREA_LABS]
SHAPE_LABS = ["N/A", "focus", "round", "oval", "elongated", "irregular"]
SHAPE_LABS = [add_spaces(lab) for lab in SHAPE_LABS]
SAT_LABS = ["N/A", "single lesion", "core with satellite lesions", "scattered lesions"]
SAT_LABS = [add_spaces(lab) for lab in SAT_LABS]
LOBES = ["n/a", "frontal", "parietal", "occipital", "temporal", "limbic",
         "insula", "subcortical", "cerebellum", "brainstem"]
LOBES = [add_spaces(lob) for lob in LOBES]

area_inv = {i + 1: lab for i, lab in enumerate(AREA_LABS)}
shape_inv = {i + 1: lab for i, lab in enumerate(SHAPE_LABS)}
satellite_inv = {i + 1: lab for i, lab in enumerate(SAT_LABS)}
lobe_inv = {i + 1: lob for i, lob in enumerate(LOBES)}


def load_jsonl(path: Union[str, Path]) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_answer(text, lbl, correct, total, remove_labels):
    total += 1
    if lbl in text:
        correct += 1
        text = text.replace(lbl, " ", 1)
    else:
        for remove_label in remove_labels:
            text = text.replace(remove_label, " ", 1)

    return text, correct, total


def check_region(text, gt_lobes, correct, total, remove_labels, calculate_accuracy=True):
    total += 1
    tp = 0
    fn = 0
    if gt_lobes == [" n/a "]:
        if " n/a " in text:
            correct += 1
            text = text.replace(" n/a ", " ", 1)
        elif " N/A " in text:
            correct += 1
            text = text.replace(" N/A ", " ", 1)
        for remove_label in remove_labels:
            text = text.replace(remove_label, " ", 1)
    else:
        for lob_lbl in gt_lobes:
            if lob_lbl in text:
                text = text.replace(lob_lbl, " ", 1)
                tp += 1
            else:
                fn += 1
        # penalise for extra lobes not in GT
        extraneous = [lbl for lbl in lobe_inv.values() if (lbl not in gt_lobes) and (lbl in text)]
        fp = len(extraneous)
        if calculate_accuracy:
            tn = len(LOBES) - (tp + fp + fn)
            acc = (tp+tn) / (tp + tn + fp + fn)
            correct += acc
        else:
            iou = tp / (tp + fp + fn)
            correct += iou
        for remove_label in remove_labels:
            text = text.replace(remove_label, " ", 1)
    return text, correct, total


def check_for_no_labels(text, correct, total, check_labels, ignore_labels):
    total += 1
    check_labels = [lbl for lbl in check_labels if lbl not in ignore_labels]
    if all(lbl not in text for lbl in check_labels):
        correct += 1
    return correct, total


def collect_model_answer(pred_d):
    if "model_answer" in pred_d:
        text = pred_d["model_answer"]
    elif "pred" in pred_d:
        text = pred_d["pred"]
    else:
        raise ValueError("No prediction found in JSON!")
    return text


def collect_gt_answer(answer_d):
    return answer_d["answer"]


def evaluate_pair(gt_d, pred_d, correct, total, include_nonzero=False, calculate_accuracy=False):
    """Grade one GT / prediction pair and update counters."""
    gt_ans = gt_d["answer_vqa_numeric"]
    text = collect_model_answer(pred_d)
    text = add_spaces(text)

    area_gt = gt_ans[0]
    if area_gt != 0:
        lbl = area_inv[area_gt]
        text, correct["area"], total["area"] = check_answer(text=text, lbl=lbl, correct=correct["area"],
                                                            total=total["area"], remove_labels=AREA_LABS)
    else:
        if include_nonzero:
            correct["area"], total["area"] = check_for_no_labels(text=text, correct=correct["area"],
                                                                 total=total["area"], check_labels=AREA_LABS,
                                                                 ignore_labels=[" N/A "])
    region_gt = gt_ans[1]
    if region_gt not in (0, [0]):
        gt_lobes = [lobe_inv[lob] for lob in region_gt]
        text, correct["region"], total["region"] = check_region(text=text, gt_lobes=gt_lobes, correct=correct["region"],
                                                                total=total["region"], remove_labels=LOBES,
                                                                calculate_accuracy=calculate_accuracy)
    else:
        if include_nonzero:
            correct["region"], total["region"] = check_for_no_labels(text=text, correct=correct["region"],
                                                                     total=total["region"], check_labels=LOBES,
                                                                     ignore_labels=[" N/A "])
    shape_gt = gt_ans[2]
    if shape_gt != 0:
        lbl = shape_inv[shape_gt]
        text, correct["shape"], total["shape"] = check_answer(text=text, lbl=lbl, correct=correct["shape"],
                                                              total=total["shape"], remove_labels=SHAPE_LABS)
    else:
        if include_nonzero:
            correct["shape"], total["shape"] = check_for_no_labels(text=text, correct=correct["shape"],
                                                                   total=total["shape"], check_labels=SHAPE_LABS,
                                                                   ignore_labels=[" N/A "])
    sat_gt = gt_ans[3]
    if sat_gt != 0:
        lbl = satellite_inv[sat_gt]
        text, correct["satellite"], total["satellite"] = check_answer(text=text, lbl=lbl, correct=correct["satellite"],
                                                                      total=total["satellite"], remove_labels=SAT_LABS)
    else:
        if include_nonzero:
            correct["satellite"], total["satellite"] = check_for_no_labels(text=text, correct=correct["satellite"],
                                                                           total=total["satellite"], check_labels=SAT_LABS,
                                                                           ignore_labels=[" N/A "])

    unk_gt = gt_ans[4]
    if unk_gt != 0:
        correct["unknown"], total["unknown"] = check_for_no_labels(text=text, correct=correct["unknown"],
                                                                   total=total["unknown"],
                                                                   check_labels=AREA_LABS+LOBES+SHAPE_LABS+SAT_LABS,
                                                                   ignore_labels=[])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file",   required=True, help="Ground‑truth JSON file")
    parser.add_argument("--pred_file", required=True, help="Predictions JSON file")
    parser.add_argument("--out_file", required=True, help="Where to write aggregated JSON")
    parser.add_argument("--include_nonzero", action="store_true", default=False, required=False)
    parser.add_argument("--calculate_accuracy", action="store_true", default=False, required=False)
    parser.add_argument("--evaluate_fluency", action="store_true", default=False, required=False)
    parser.add_argument("--bert_model", type=str, default="microsoft/deberta-xlarge-mnli", help="HuggingFace model to use for BERTScore")
    args = parser.parse_args()

    gt = load_jsonl(args.gt_file)
    pred = load_jsonl(args.pred_file)
    assert len(gt) == len(pred), "GT and prediction files differ in length!"

    correct, total = Counter(), Counter()
    for g, p in zip(gt, pred):
        evaluate_pair(g, p, correct, total, include_nonzero=args.include_nonzero,
                      calculate_accuracy=args.calculate_accuracy)

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

    if args.evaluate_fluency:
        if bert_score is None:
            raise ImportError("bert_score not installed — run `pip install bert_score`")
        cand_texts = [collect_model_answer(pred_d) for pred_d in pred]
        ref_texts = [collect_gt_answer(gt_d) for gt_d in gt]
        P, R, F1 = bert_score(cand_texts, ref_texts, model_type=args.bert_model, lang="en", rescale_with_baseline=True)
        results["bert_score_precision"] = P.mean().item()
        results["bert_score_recall"] = R.mean().item()
        results["bert_score_f1"] = F1.mean().item()

        print("\nBERTScore (fluency / semantic adequacy)")
        print("---------------------------------------")
        print(f"P:  {results['bert_score_precision']:.4f}")
        print(f"R:  {results['bert_score_recall']:.4f}")
        print(f"F1: {results['bert_score_f1']:.4f}")
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
