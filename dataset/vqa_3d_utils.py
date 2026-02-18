from skimage import measure
import numpy as np
import os
import json
from scipy import ndimage
from scipy.ndimage import center_of_mass
from nilearn.masking import compute_brain_mask
from skimage.morphology import ball
import nibabel as nib
import pandas as pd
from localize_brain import localize_to_brain_regions, load_aal_atlas_label_map, get_region_str
from vqa_utils import compute_area_percentage, vqa_round, label_names, goat_label_names, ped_label_names


area_map = {lab.lower(): i+1 for i, lab in enumerate(
    ["N/A", "<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-75%"])} # start at 1


shape_map = {lab.lower(): i+1 for i, lab in enumerate(
    ["N/A", "focus", "round", "oval", "elongated", "irregular"])} # start at 1


satellite_map = {lab.lower(): i+1 for i, lab in enumerate(
    ["N/A", "single lesion", "core with satellite lesions", "scattered lesions"])} # start at 1


lobes = ["n/a", "frontal", "parietal", "occipital", "temporal",
         "limbic", "insula", "subcortical", "cerebellum", "brainstem"]
lobe_map = {lob: i+1 for i, lob in enumerate(lobes)}  # start at 1


def region_to_codes(region):
    """Return sorted list of lobe indices (empty list == 0)."""
    txt = region.strip().lower()
    parts = []
    for chunk in txt.split(","):
        parts += [p.strip() for p in chunk.split(" and ")]      # handle “and”
    if len(parts) == 0:
        return [0]
    return sorted({lobe_map[p] for p in parts if p in lobe_map})


def answer_to_numeric(task_idx: int, ans: str):
    ans = ans.lower().strip()
    if task_idx == 1:  # area
        return area_map.get(ans, 0)
    if task_idx == 2:  # region
        return region_to_codes(ans)
    if task_idx == 3:  # shape
        return shape_map.get(ans, 0)
    if task_idx == 4:  # satellite
        return satellite_map.get(ans, 0)
    if task_idx == 5:  # unknown
        # hard code return 1 if the task id is passed
        return 1
    return 0


def convert_entry(entry: dict, add_unknown=False):
    """
    Build [area, region(list), shape, satellite, unknown, unknown] for a single VQA entry.
    Defaults: 0 for scalar tasks, [] for region when task not answered.
    """
    numeric = [0, [0], 0, 0]                     # area, region, shape, satellite
    if add_unknown:
        numeric = numeric + [0]
    for task_idx, answers in zip(entry["combo"], entry["answer_vqa"]):
        if answers:                             # answers is a 1‑element list
            numeric[task_idx - 1] = answer_to_numeric(task_idx, answers[0])
    return numeric


def summarise_vqa_stats(vqa_list):
    """
    vqa_list : list[dict] produced by your VQA‑generation pipeline

    Returns
    -------
    {
      "total_questions"                 : int,
      "questions_per_label"             : pd.Series
      "questions_per_type"              : pd.Series
      "questions_per_label_and_type"    : pd.DataFrame   (labels × types)

      "answer_dist_per_type"            : {type : DataFrame}               # NEW
      "answer_dist_per_label_and_type"  : {type : DataFrame}               # NEW
    }
    """
    df = pd.json_normalize(vqa_list)
    df["answer"] = df["answer_vqa"].str[0]      # unwrap the single‑item list

    # ------------------------------------------------------------------ #
    # 1. Basic question counts
    # ------------------------------------------------------------------ #
    total_questions              = len(df)
    questions_per_label          = df["label_name"].value_counts()
    questions_per_type           = df["type"].value_counts()
    questions_per_label_and_type = (
        df.groupby(["label_name", "type"])
          .size()
          .unstack(fill_value=0)      # labels as rows, types as columns
    )

    # ------------------------------------------------------------------ #
    # 2. Answer‑distribution tables, **one DataFrame per question‑type**
    # ------------------------------------------------------------------ #
    answer_dist_per_type = {}
    answer_dist_per_label_and_type = {}

    for qtype, sub in df.groupby("type"):
        # a) overall distribution for this type  -----------------------
        #     columns = unique answers, single row with counts
        overall_tbl = (
            sub.groupby("answer")
               .size()
               .rename("count")
               .to_frame()
               .T                          # single row
               .fillna(0)
               .astype(int)
        )
        answer_dist_per_type[qtype] = overall_tbl

        # b) distribution broken down by label ------------------------
        #     rows = label_name, columns = answers
        per_label_tbl = (
            sub.groupby(["label_name", "answer"])
               .size()
               .unstack(fill_value=0)
               .astype(int)
        )
        answer_dist_per_label_and_type[qtype] = per_label_tbl

    # ------------------------------------------------------------------ #
    # 3. Package everything
    # ------------------------------------------------------------------ #
    return {
        "total_questions": total_questions,
        "questions_per_label": questions_per_label,
        "questions_per_type": questions_per_type,
        "questions_per_label_and_type": questions_per_label_and_type,
        "answer_dist_per_type": answer_dist_per_type,                       # dict[type] → DF
        "answer_dist_per_label_and_type": answer_dist_per_label_and_type,   # dict[type] → DF
    }

def postprocess_3d_vqa_data(all_vqa_questions, save_vqa_file="brats_gli_vqa_clean_data.json", seed=0):
    for index in range(len(all_vqa_questions)):
        question = all_vqa_questions[index]
        question["img_id"] = all_vqa_questions[index]["volume_file_id"]
        assert "question" in all_vqa_questions[index]
        question["q_lang"] = "en"
        question["qid"] = index
        question["location"] = "Brain"
        question["answer_type"] = "OPEN"
        question["base_type"] = "VQA"
        question["content_type"] = question["type"]
        question["qid"] = index
        base_dir = os.path.basename(question["volume_file_dir"])
        if "gli" in base_dir.lower() or "met" in base_dir.lower() or "ped" in base_dir.lower():
            question["study_name"] = "-".join(base_dir.split("-")[:-1])
        elif "goat" in base_dir.lower():
            question["study_name"] = base_dir
        else:
            raise ValueError(f"Unknown study name: {base_dir}")
        question["volume_non_seg_files"]['t1c'] = question["volume_non_seg_files"]['t1c'].replace("t1c_MNI.nii.gz",
                                                                                                  "t1c.nii.gz")

    with open(save_vqa_file, 'w') as f:
        json.dump(all_vqa_questions, f, indent=2)

    return all_vqa_questions


def generate_3d_labal_vqa_questions(summ, include_area=True, include_quadrant=True, include_bbox=True,
                                    include_extent=True, include_solidity=True, subjective_only=False):
    vqa_questions = []
    if include_area:
        question = f"How large is the volume covered by {summ['name']}?"
        if subjective_only:
            answer = f"{summ['area_interp']}"
        else:
            answer = f"{summ['area_pct']:.1f}%, which is {summ['area_interp']}"
        question_dict = {"question": question, "answer": answer, "type": "area", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_quadrant:
        question = f"Which quadrant is {summ['name']} centered in?"
        answer = f"{summ['centroid_quadrant']}"
        question_dict = {"question": question, "answer": answer, "type": "quadrant", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_bbox:
        question = f"The smallest bounding cube surrounding {summ['name']} is in which quadrants?"
        answer = f"{summ['bbox_str']}"
        question_dict = {"question": question, "answer": answer, "type": "bbox", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_extent:
        question = f"Within the smallest bounding cube surrounding {summ['name']}, to what extent is the bounding cube region filled?"
        if subjective_only:
            answer = f"{summ['extent_interp']}"
        else:
            answer = f"{summ['extent_value']:.1f}%, which is {summ['extent_interp']}"
        question_dict = {"question": question, "answer": answer, "type": "extent", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_solidity:
        question = f"How compact is the {summ['name']} region?"
        if subjective_only:
            answer = f"{summ['solidity_interp']}"
        else:
            answer = f"{summ['solidity_value']:.1f}%, which is {summ['solidity_interp']}"
        question_dict = {"question": question, "answer": answer, "type": "solidity", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    return vqa_questions


def generate_3d_labal_vqa_questions_v2(
    summ,
    include_area=True,
    include_bbox=True,
    include_extent=True,
    include_solidity=True,
    include_area_bbox=True,
    include_area_extent=True,
    include_area_solidity=True,
    include_bbox_extent=True,
    include_bbox_solidity=True,
    include_extent_solidity=True,
    include_area_bbox_extent=True,
    include_area_bbox_solidity=True,
    include_bbox_extent_solidity=True,
    include_area_bbox_extent_solidity=True
):
    vqa_questions = []

    # 1) AREA (example done)
    if include_area:
        question = f"How large is the volume covered by {summ['name']}?"
        # short VQA
        answer_vqa = [summ['area_interp']]
        # longer, fluent text
        answer_gen = f"The overall volume of {summ['name']} is {summ['area_interp']}."
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "area",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 2) BBOX (example done)
    if include_bbox:
        question = f"The smallest bounding cube surrounding {summ['name']} is in which quadrants?"
        answer_vqa = [summ['bbox_str']]
        answer_gen = f"The bounding region for {summ['name']} spans {summ['bbox_str']} in the image space."
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "bbox",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 3) EXTENT (example done)
    if include_extent:
        question = f"Within the smallest bounding cube surrounding {summ['name']}, to what extent is the bounding cube region filled?"
        answer_vqa = [summ['extent_interp']]
        answer_gen = f"Inside its bounding region, {summ['name']} occupies {summ['extent_interp']} of that cube."
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "extent",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 4) SOLIDITY
    if include_solidity:
        question = f"How compact is the {summ['name']} region?"
        answer_vqa = [summ['solidity_interp']]
        answer_gen = f"Based on its shape analysis, {summ['name']} is {summ['solidity_interp']} in terms of compactness."
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "solidity",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 5) AREA + BBOX
    if include_area_bbox:
        question = f"How large is the volume of {summ['name']}, and in which quadrants does its smallest bounding cube lie?"
        answer_vqa = [summ['area_interp'], summ['bbox_str']]
        answer_gen = (
            f"The volume of {summ['name']} is {summ['area_interp']}, and its bounding cube lies in {summ['bbox_str']}."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "area_bbox",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 6) AREA + EXTENT
    if include_area_extent:
        question = f"How large is the volume of {summ['name']}, and how much of its bounding cube is filled?"
        answer_vqa = [summ['area_interp'], summ['extent_interp']]
        answer_gen = (
            f"The overall volume of {summ['name']} is {summ['area_interp']}, "
            f"and it fills {summ['extent_interp']} of its bounding cube."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "area_extent",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 7) AREA + SOLIDITY
    if include_area_solidity:
        question = f"How large is the volume of {summ['name']}, and how compact would you describe that region to be?"
        answer_vqa = [summ['area_interp'], summ['solidity_interp']]
        answer_gen = (
            f"The volume of {summ['name']} is {summ['area_interp']}, "
            f"and it appears {summ['solidity_interp']} in terms of compactness."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "area_solidity",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 8) BBOX + EXTENT
    if include_bbox_extent:
        question = (
            f"What are the quadrants for the smallest bounding cube surrounding {summ['name']}, "
            f"and to what extent is the bounding cube region filled?"
        )
        answer_vqa = [summ['bbox_str'], summ['extent_interp']]
        answer_gen = (
            f"The bounding cube is located in {summ['bbox_str']}, and {summ['name']} occupies "
            f"{summ['extent_interp']} of that region."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "bbox_extent",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 9) BBOX + SOLIDITY
    if include_bbox_solidity:
        question = f"What are the quadrants for the smallest bounding cube surrounding {summ['name']}, and how compact is the region?"
        answer_vqa = [summ['bbox_str'], summ['solidity_interp']]
        answer_gen = (
            f"The bounding cube is in {summ['bbox_str']}, and {summ['name']} shows "
            f"{summ['solidity_interp']} compactness."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "bbox_solidity",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 10) EXTENT + SOLIDITY
    if include_extent_solidity:
        question = f"To what extent is the bounding cube region filled and how compact is the {summ['name']} region?"
        answer_vqa = [summ['extent_interp'], summ['solidity_interp']]
        answer_gen = (
            f"{summ['name']} occupies {summ['extent_interp']} of its bounding cube, and "
            f"it is {summ['solidity_interp']} in shape."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "extent_solidity",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 11) AREA + BBOX + EXTENT
    if include_area_bbox_extent:
        question = (
            f"How large is the volume covered by {summ['name']}, what are the quadrants for the smallest "
            f"bounding cube surrounding it, and to what extent is the bounding cube region filled?"
        )
        answer_vqa = [summ['area_interp'], summ['bbox_str'], summ['extent_interp']]
        answer_gen = (
            f"The volume of {summ['name']} is {summ['area_interp']}. Its bounding cube spans {summ['bbox_str']}, "
            f"and the region fills {summ['extent_interp']} of that cube."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "area_bbox_extent",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 12) AREA + BBOX + SOLIDITY
    if include_area_bbox_solidity:
        question = (
            f"How large is the volume covered by {summ['name']}, what is the smallest bounding cube surrounding it, "
            f"and how compact is the region?"
        )
        answer_vqa = [summ['area_interp'], summ['bbox_str'], summ['solidity_interp']]
        answer_gen = (
            f"The volume of {summ['name']} is {summ['area_interp']}, its bounding cube lies in {summ['bbox_str']}, "
            f"and the region appears {summ['solidity_interp']} in terms of compactness."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "area_bbox_solidity",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 13) AREA + EXTENT + SOLIDITY
    if include_area_bbox_solidity:
        question = (
            f"How large is the volume covered by {summ['name']}, to what extent is its bounding cube filled, "
            f"and how compact is the region?"
        )
        answer_vqa = [summ['area_interp'], summ['extent_interp'], summ['solidity_interp']]
        answer_gen = (
            f"The volume of {summ['name']} is {summ['area_interp']}, the label fills {summ['extent_interp']} "
            f"of its bounding cube ",
            f"and the region appears {summ['solidity_interp']} in terms of compactness."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "area_extent_solidity",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 14) BBOX + EXTENT + SOLIDITY
    if include_bbox_extent_solidity:
        question = (
            f"What are the quadrants for the smallest bounding cube surrounding {summ['name']}, "
            f"to what extent is it filled, and how compact is the region?"
        )
        answer_vqa = [summ['bbox_str'], summ['extent_interp'], summ['solidity_interp']]
        answer_gen = (
            f"The bounding cube for {summ['name']} is in {summ['bbox_str']}, the label fills {summ['extent_interp']} "
            f"of that cube, and it is {summ['solidity_interp']} overall."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "bbox_extent_solidity",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    # 15) AREA + BBOX + EXTENT + SOLIDITY
    if include_area_bbox_extent_solidity:
        question = (
            f"How large is the volume covered by {summ['name']}, what are the quadrants for the smallest bounding cube, "
            f"to what extent is that cube filled, and how compact is the region?"
        )
        answer_vqa = [summ['area_interp'], summ['bbox_str'], summ['extent_interp'], summ['solidity_interp']]
        answer_gen = (
            f"The volume of {summ['name']} is {summ['area_interp']}. Its bounding cube spans {summ['bbox_str']}, "
            f"the region fills {summ['extent_interp']} of that space, and it is {summ['solidity_interp']} in shape."
        )
        question_dict = {
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "area_bbox_extent_solidity",
            "label_name": summ['name']
        }
        vqa_questions.append(question_dict)

    return vqa_questions


def generate_3d_labal_vqa_questions_v3(
    summ,
    include_area=True,
    include_regions=True,
    include_shape=True,
    include_satellite=True,
):
    vqa_questions = []

    # Single attributes
    if include_area:
        question = f"How large is the volume covered by {summ['name']}?"
        answer_vqa = [summ['area_interp']]
        answer_gen = f"The overall volume of {summ['name']} is {summ['area_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "area",
            "label_name": summ['name']
        })

    if include_regions:
        question = f"Which region(s) of the brain is {summ['name']} located in?"
        answer_vqa = [summ['regions']]
        answer_gen = f"The {summ['name']} is located in {summ['regions']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "region",
            "label_name": summ['name']
        })

    if include_shape:
        question = f"What is the shape of {summ['name']}?"
        answer_vqa = [summ['shape_interp']]
        answer_gen = f"The shape of {summ['name']} is {summ['shape_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "shape",
            "label_name": summ['name']
        })

    if include_satellite:
        question = f"How spread out is {summ['name']}?"
        answer_vqa = [summ['satellite_interp']]
        answer_gen = f"The spread of {summ['name']} is {summ['satellite_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": answer_vqa,
            "answer_gen": answer_gen,
            "type": "satellite",
            "label_name": summ['name']
        })
    """
    # 2-Way Combinations
    if include_area and include_regions:
        question = f"How large is the volume of {summ['name']} and where is it located?"
        answer_gen = f"The overall volume of {summ['name']} is {summ['area_interp']}, and it is located in {summ['regions']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['area_interp'], summ['regions']],
            "answer_gen": answer_gen,
            "type": "area_region",
            "label_name": summ['name']
        })

    if include_area and include_shape:
        question = f"How large is the volume of {summ['name']} and what is its shape?"
        answer_gen = f"The overall volume of {summ['name']} is {summ['area_interp']}, and its shape is described as {summ['shape_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['area_interp'], summ['shape_interp']],
            "answer_gen": answer_gen,
            "type": "area_shape",
            "label_name": summ['name']
        })

    if include_area and include_satellite:
        question = f"How large is the volume of {summ['name']} and how spread out is it?"
        answer_gen = f"The overall volume of {summ['name']} is {summ['area_interp']}, and it is characterized as {summ['satellite_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['area_interp'], summ['satellite_interp']],
            "answer_gen": answer_gen,
            "type": "area_satellite",
            "label_name": summ['name']
        })

    if include_regions and include_shape:
        question = f"In which region is {summ['name']} and what is its shape?"
        answer_gen = f"The {summ['name']} is located in {summ['regions']}, and its shape is described as {summ['shape_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['regions'], summ['shape_interp']],
            "answer_gen": answer_gen,
            "type": "region_shape",
            "label_name": summ['name']
        })

    if include_regions and include_satellite:
        question = f"In which region is {summ['name']} and how spread out is it?"
        answer_gen = f"The {summ['name']} is located in {summ['regions']}, and it is characterized as {summ['satellite_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['regions'], summ['satellite_interp']],
            "answer_gen": answer_gen,
            "type": "region_satellite",
            "label_name": summ['name']
        })

    if include_shape and include_satellite:
        question = f"What is the shape of {summ['name']} and how spread out is it?"
        answer_gen = f"The shape of {summ['name']} is described as {summ['shape_interp']}, and it is characterized as {summ['satellite_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['shape_interp'], summ['satellite_interp']],
            "answer_gen": answer_gen,
            "type": "shape_satellite",
            "label_name": summ['name']
        })

    # 3-Way Combinations
    if include_area and include_regions and include_shape:
        question = f"What is the volume, region, and shape of {summ['name']}?"
        answer_gen = f"The overall volume of {summ['name']} is {summ['area_interp']}, it is located in {summ['regions']}, and its shape is described as {summ['shape_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['area_interp'], summ['regions'], summ['shape_interp']],
            "answer_gen": answer_gen,
            "type": "area_region_shape",
            "label_name": summ['name']
        })

    if include_area and include_shape and include_satellite:
        question = f"What is the volume, shape, and spread of {summ['name']}?"
        answer_gen = f"The overall volume of {summ['name']} is {summ['area_interp']}, its shape is {summ['shape_interp']}, and it is characterized as {summ['satellite_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['area_interp'], summ['shape_interp'], summ['satellite_interp']],
            "answer_gen": answer_gen,
            "type": "area_shape_satellite",
            "label_name": summ['name']
        })

    if include_regions and include_shape and include_satellite:
        question = f"What is the region, shape, and spread of {summ['name']}?"
        answer_gen = f"The {summ['name']} is located in {summ['regions']}, its shape is described as {summ['shape_interp']}, and it is characterized as {summ['satellite_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['regions'], summ['shape_interp'], summ['satellite_interp']],
            "answer_gen": answer_gen,
            "type": "region_shape_satellite",
            "label_name": summ['name']
        })

    # 4-Way Combination
    if include_area and include_regions and include_shape and include_satellite:
        question = f"What is the volume, region, shape, and spread of {summ['name']}?"
        answer_gen = f"The overall volume of {summ['name']} is {summ['area_interp']}, it is located in {summ['regions']}, its shape is described as {summ['shape_interp']}, and it is characterized as {summ['satellite_interp']}."
        vqa_questions.append({
            "question": question,
            "answer_vqa": [summ['area_interp'], summ['regions'], summ['shape_interp'], summ['satellite_interp']],
            "answer_gen": answer_gen,
            "type": "area_region_shape_satellite",
            "label_name": summ['name']
        })
    """
    return vqa_questions


def analyze_3d_label_summary(nib_seg_map_3d, seg_map_3d, nib_t1n_3d, height, width, depth, total_pixels,
                             labels_order=(1, 2, 3, 4), aal_version="SPM12", pediatric=False, goat=False):
    """
    For each label (1..4), compute:
      - area percentage + subjective interpretation
      - centroid quadrant
      - bounding box quadrants
      - extent-based "compactness" measure
    """
    label_summaries = []
    atlas_img, atlas_label_map = load_aal_atlas_label_map(version=aal_version)
    atlas_img = nib.as_closest_canonical(atlas_img)

    for lbl in labels_order:
        summ = {}
        # TODO: Fix for Tumor Core (if we use it)
        mask = seg_map_3d == lbl
        if goat:
            label_name = goat_label_names.get(lbl, f"Label {lbl}")
        elif pediatric:
            label_name = ped_label_names.get(lbl, f"Label {lbl}")
        else:
            label_name = label_names.get(lbl, f"Label {lbl}")
        area_pct = compute_area_percentage_v1(mask, nib_t1n_3d)
        area_interp = interpret_3d_area_percentage(area_pct)
        if area_interp == "N/A":
            summ['satellite_interp'] = "N/A"
            summ['shape_interp'] = "N/A"
            summ['regions'] = "N/A"
        else:
            regions = localize_to_brain_regions(nib_seg_map_3d, atlas_img, atlas_label_map, label_index=lbl)['regions']
            region_str = get_region_str(regions)
            summ['regions'] = region_str
            summ.update(compute_shape_descriptors(mask))
        """
        if area_interp == "none":
            centroid = None
            quadrant = "none"
            bounding_box_quads = None
            bounding_box_str = "none"
            extent_value = 0.0
            extent_interp = "none"
            solidity_value = 0.0
            solidity_interp = "none"
        else:
            centroid = center_of_mass(mask)
            quadrant = get_3d_quadrant(centroid, height, width, depth)

            bbox = compute_3d_bounding_box(mask, total_pixels)
            bounding_box_quads = get_3d_bounding_box_quadrants(bbox, height, width, depth)
            bounding_box_str = bounding_box_quads if bounding_box_quads else "none"

            # Extent-based compactness
            extent_value, extent_interp = measure_3d_extent_compactness(mask, bbox)
            solidity_value, solidity_interp = measure_3d_solidity(mask)

        if (bounding_box_str == "none") or (extent_interp == "none") or (solidity_interp == "none"):
            centroid = None
            quadrant = "none"
            bounding_box_quads = None
            bounding_box_str = "none"
            extent_value = 0.0
            extent_interp = "none"
            solidity_value = 0.0
            solidity_interp = "none"

        summ.update({
            "label": lbl,
            "name": label_name,
            "area_pct": area_pct,
            "area_interp": area_interp,
            "centroid_quadrant": quadrant,
            "bbox_quadrants": bounding_box_quads,
            "bbox_str": bounding_box_str,
            "extent_value": extent_value,
            "extent_interp": extent_interp,
            "solidity_value": solidity_value,
            "solidity_interp": solidity_interp
        })
        """
        summ.update({
            "label": lbl,
            "name": label_name,
            "area_pct": area_pct,
            "area_interp": area_interp,
        })
        label_summaries.append(summ)
    return label_summaries


def _surface_area_from_mesh(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()

def _metrics_for_component(comp_mask, voxel_vol, voxel_spacing):
    """Return (sphericity, elongation, flatness, solidity, compactness)."""
    V = comp_mask.sum() * voxel_vol
    if V == 0:
        return (0., 0., 0., 1., 0.)

    verts, faces, _, _ = measure.marching_cubes(
        comp_mask.astype(np.uint8), spacing=voxel_spacing
    )
    A = _surface_area_from_mesh(verts, faces)
    sphericity = (np.pi ** (1 / 3) * (6 * V) ** (2 / 3)) / A if A > 0 else 0.0
    compactness = A / V if V > 0 else 0.0

    coords = np.column_stack(np.nonzero(comp_mask))
    if coords.shape[0] >= 3:
        cov = np.cov(coords, rowvar=False)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)[::-1]
        elongation = np.sqrt(eigvals[0] / eigvals[1]) if eigvals[1] > 0 else 0.0
        flatness   = np.sqrt(eigvals[2] / eigvals[1]) if eigvals[1] > 0 else 0.0
    else:
        elongation = flatness = 0.0

    try:
        hull = measure.ConvexHull(coords)
        V_hull = hull.volume * voxel_vol
        solidity = V / V_hull if V_hull > 0 else 1.0
    except Exception:
        solidity = 1.0

    return (sphericity, elongation, flatness, solidity, compactness)
# ------------------------------------------------------------------


def compute_shape_descriptors(mask, voxel_spacing=(1., 1., 1.)):
    desc = {}
    voxel_vol = np.prod(voxel_spacing)
    total_V = mask.sum() * voxel_vol
    desc["volume_mm3"] = total_V

    labeled, num_cc = ndimage.label(mask, structure=ball(1))
    desc["multiplicity"] = num_cc

    if num_cc == 0:                    # empty mask
        for k in ("sphericity", "elongation", "flatness",
                  "solidity", "compactness"):
            desc[k] = 0.0
        desc["metric_source"] = "none"
        desc["satellite_interp"] = "N/A"
        desc["shape_interp"] = "N/A"
        return desc

    # ----- basic component statistics -----
    cc_sizes = ndimage.sum(mask, labeled, index=range(1, num_cc + 1))
    cc_sizes = np.asarray(cc_sizes, dtype=float) * voxel_vol
    core_idx = int(np.argmax(cc_sizes)) + 1
    core_vol = cc_sizes.max()
    core_fraction = core_vol / total_V
    desc["core_fraction"] = core_fraction
    desc["satellite_ratio"] = max(0, num_cc - 1) / num_cc

    # satellite label
    if num_cc == 1:
        desc["satellite_interp"] = "single lesion"
    elif core_fraction >= 0.85:
        desc["satellite_interp"] = "core with satellite lesions"
    else:
        desc["satellite_interp"] = "scattered lesions"

    # ------------------------------------------------------------------
    # Metric extraction: use core OR mean of all components
    # ------------------------------------------------------------------
    if num_cc == 1 or core_fraction >= 0.85:
        # use the core component
        core_mask = (labeled == core_idx)
        sph, elg, flat, sol, comp = _metrics_for_component(
            core_mask, voxel_vol, voxel_spacing
        )
        desc["metric_source"] = "core"
    else:
        # scattered → average across components
        metrics = np.zeros(5)
        for cid in range(1, num_cc + 1):
            comp_mask = (labeled == cid)
            metrics += np.array(_metrics_for_component(
                comp_mask, voxel_vol, voxel_spacing
            ))
        sph, elg, flat, sol, comp = metrics / num_cc
        desc["metric_source"] = "mean"

    desc.update({
        "sphericity": sph,
        "elongation": elg,
        "flatness": flat,
        "solidity": sol,
        "compactness": comp
    })

    # ----- shape label -----
    if total_V * 1e-3 < 0.1:
        shape_word = "focus"
    else:
        if sph >= 0.80 and elg <= 1.4:
            shape_word = "round"
        elif 0.50 <= sph < 0.80 and 1.4 < elg <= 3.0:
            shape_word = "oval"
        elif elg > 3.0:
            shape_word = "elongated"
        else:
            shape_word = "irregular"
    desc["shape_interp"] = shape_word

    return desc


def compute_3d_bounding_box(mask, total_pixels):
    """
    Returns (min_row, min_col, max_row, max_col) for all True pixels in `mask`.
    If `mask` is empty, returns None.
    """
    area = compute_area_percentage(mask, total_pixels)
    coords = np.where(mask)
    if (coords[0].size == 0) or (area == 0.0):
        return None
    min_r, max_r = coords[0].min(), coords[0].max() + 1
    min_c, max_c = coords[1].min(), coords[1].max() + 1
    min_d, max_d = coords[2].min(), coords[2].max() + 1
    return min_r, min_c, min_d, max_r, max_c, max_d


def interpret_3d_area_percentage(pct: float) -> str:
    """
    Map percentage to textual bin.
    Uses literature‑like cut‑offs: <1, 1–5, 5–10, 10–25, 25–50, 50–75, >75.
    """
    if pct == 0.0:
        return "N/A"
    if pct < 1.0:
        return "<1%"
    if pct < 5.0:
        return "1-5%"
    if pct < 10.0:
        return "5-10%"
    if pct < 25.0:
        return "10-25%"
    if pct < 50.0:
        return "25-50%"
    if pct < 75.0:
        return "50-75%"
    return ">75%"


def get_3d_quadrant(centroid, height, width, depth):
    """
    Maps a (row, col) centroid to one of 27 quadrants (top-left to bottom-right).
    """
    if not centroid or np.isnan(centroid[0]) or np.isnan(centroid[1]):
        return "none"

    row, col, d = centroid
    third_height = height / 3
    third_width = width / 3
    third_depth = depth / 3
    quadrant_string = ""
    # add row string
    if row < third_height:
        quadrant_string += "top-"
    elif row < 2 * third_height:
        quadrant_string += "center-"
    else:
        quadrant_string += "bottom-"
    # add col string
    if col < third_width:
        quadrant_string += "left-"
    elif col < 2 * third_width:
        quadrant_string += "center-"
    else:
        quadrant_string += "right-"
    # add depth string
    if d < third_depth:
        quadrant_string += "front"
    elif d < 2 * third_depth:
        quadrant_string += "middle"
    else:
        quadrant_string += "back"
    return quadrant_string


def get_3d_bounding_box_quadrants(bbox, height, width, depth):
    """
    Determine which quadrants are affected by a bounding box
    by sampling corners of the bounding box.
    """
    if not bbox:
        return "none"

    min_r, min_c, min_d, max_r, max_c, max_d = bbox
    affected_quadrants = set()

    # We check the eight corners
    corners = [
        (min_r, min_c, min_d),
        (min_r, max_c - 1, min_d),
        (max_r - 1, min_c, min_d),
        (max_r - 1, max_c - 1, min_d),
        (min_r, min_c, max_d - 1),
        (min_r, max_c - 1, max_d - 1),
        (max_r - 1, min_c, max_d - 1),
        (max_r - 1, max_c - 1, max_d - 1),
    ]
    for (r, c, d) in corners:
        quadrant = get_3d_quadrant((r, c, d), height, width, depth)
        if quadrant != "none":
            affected_quadrants.add(quadrant)
    if len(affected_quadrants) == 0:
        return "none"
    affected_quadrants_str = ", ".join(sorted(affected_quadrants))
    return affected_quadrants_str


def measure_3d_extent_compactness(mask, bbox):
    area = mask.sum()
    if not bbox:
        return 0.0, "none"

    min_r, min_c, min_d, max_r, max_c, max_d = bbox
    bbox_h = max_r - min_r
    bbox_w = max_c - min_c
    bbox_d = max_d - min_d

    bbox_area = bbox_h * bbox_w * bbox_d
    if bbox_area == 0:
        return 0.0, "none"

    extent = (area / bbox_area) * 100
    interpretation = interpret_3d_extent(extent)
    return extent, interpretation


def interpret_3d_extent(value):
    """
    Subjective interpretation of how well the region fills its bounding box.
    """
    if value == 0.0:
        return "none"
    elif value < 5.0:
        return "very sparse"
    elif value < 12.5:
        return "somewhat scattered"
    elif value < 20.0:
        return "partially filled"
    elif value < 50.0:
        return "nearly filled"
    else:
        return "almost fully filled"


def measure_3d_solidity(mask_3d, voxel_spacing=(1.0, 1.0, 1.0)):
    # 1) Volume = number of foreground voxels * voxel volume
    voxel_volume = np.prod(voxel_spacing)  # e.g. 1 * 1 * 1 if spacing=(1,1,1)
    volume = np.count_nonzero(mask_3d) * voxel_volume
    if volume == 0:
        return 0.0, interpret_3d_solidity(0.0)

    # 2) Use marching cubes to get a 3D mesh of the surface
    #    skimage.measure.marching_cubes returns:
    #       vertices, faces, normals, values
    #    'level=0.5' is typical for binary masks
    #    'spacing' uses voxel_spacing to scale the mesh in real units.
    verts, faces, normals, _ = measure.marching_cubes(
        volume=mask_3d,
        level=0.5,
        spacing=voxel_spacing
    )

    # 3) Compute surface area of that mesh
    #    skimage provides a convenience function
    surface_area = measure.mesh_surface_area(verts, faces)

    if surface_area == 0:
        solidity = 0.0
    else:
        solidity = vqa_round(((1.6 - (surface_area / volume))/1.6) * 100)
    return solidity, interpret_3d_solidity(solidity)



def interpret_3d_solidity(value):
    """
    Subjective interpretation of solidity.
    """
    if value == 0.0:
        return "none"
    elif value < 50.0:
        return "highly irregular and scattered"
    elif value < 80.0:
        return "somewhat compact but irregular"
    else:
        return "mostly compact"


def measure_3d_extent_compactness(mask, bbox):
    area = mask.sum()
    if not bbox:
        return 0.0, "none"

    min_r, min_c, min_d, max_r, max_c, max_d = bbox
    bbox_h = max_r - min_r
    bbox_w = max_c - min_c
    bbox_d = max_d - min_d

    bbox_area = bbox_h * bbox_w * bbox_d
    if bbox_area == 0:
        return 0.0, "none"

    extent = (area / bbox_area) * 100
    interpretation = interpret_3d_extent(extent)
    return extent, interpretation


def compute_area_percentage_v1(mask, t1_n_3d, thr=1e-6):
    """
    Returns the percentage of 'mask' pixels relative to the total segmentation size.
    """
    # NOTE: This should not be necessary because all non-zero voxels are brain voxels.
    #brain_mask = compute_brain_mask(t1_n_3d).get_fdata()
    #total_pixels = brain_mask.sum()
    data = t1_n_3d.get_fdata()
    brain_mask = data > thr
    total_pixels = brain_mask.sum()
    if total_pixels == 0:
        return 0.0
    return vqa_round((mask.sum() / total_pixels) * 250)

