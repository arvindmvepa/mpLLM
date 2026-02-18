from skimage.morphology import convex_hull_image
from skimage.measure import label
import numpy as np
import re
import os
import json
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as label_, binary_dilation, generate_binary_structure
from collections import Counter, defaultdict


label_names = {
            1: "Non-Enhancing Tumor",
            2: "Surrounding Non-enhancing FLAIR hyperintensity",
            3: "Enhancing Tissue",
            4: "Resection Cavity",
            5: "Tumor Core"}

ped_label_names = {
            1: "Enhancing Tissue",
            2: "Non-Enhancing Tumor",
            3: "Cystic Component",
            4: "Peritumoral Edema"}

goat_label_names = {
            1: "Necrosis",
            2: "Edema/Invaded Tissue",
            3: "Enhancing Tumor"}



def generate_train_val_test_splits(all_vqa_questions, question_key="seg_id", seed=0, train_seg_ids=None,
                                   val_seg_ids=None, test_seg_ids=None, train_frac=0.8, val_frac=0.1,
                                   train_file="brats_gli_vqa_train.json", val_file="brats_gli_vqa_val.json",
                                   test_file="brats_gli_vqa_test.json", ignore_label="Tumor Core"):
    if ignore_label is not None:
        all_vqa_questions = [q for q in all_vqa_questions if ignore_label not in q["label_name"]]
    if (train_seg_ids is not None) and (val_seg_ids is not None) and (test_seg_ids is not None):
        train_questions = [q for q in all_vqa_questions if q[question_key] in train_seg_ids]
        val_questions = [q for q in all_vqa_questions if q[question_key] in val_seg_ids]
        test_questions = [q for q in all_vqa_questions if q[question_key] in test_seg_ids]
        train_studies = list({q["study_name"] for q in train_questions})
        val_studies = list({q["study_name"] for q in val_questions})
        test_studies = list({q["study_name"] for q in test_questions})
    else:
        random_state = np.random.RandomState(seed)
        study_names = sorted(list({q["study_name"] for q in all_vqa_questions}))
        random_state.shuffle(study_names)
        total_studies = len(study_names)
        train_end = int(total_studies * train_frac)
        val_end = int(total_studies * (train_frac + val_frac))
        train_studies = study_names[:train_end]
        val_studies = study_names[train_end:val_end]
        test_studies = study_names[val_end:]
        train_questions = [q for q in all_vqa_questions if q["study_name"] in train_studies]
        val_questions = [q for q in all_vqa_questions if q["study_name"] in val_studies]
        test_questions = [q for q in all_vqa_questions if q["study_name"] in test_studies]
    print(f"Train studies: {len(train_studies)}, Val studies: {len(val_studies)}, Test studies: {len(test_studies)}")
    print(f"Train questions: {len(train_questions)}, Val questions: {len(val_questions)}, Test questions: {len(test_questions)}")

    with open(train_file, 'w') as f:
        json.dump(train_questions, f, indent=2)
    with open(val_file, 'w') as f:
        json.dump(val_questions, f, indent=2)
    with open(test_file, 'w') as f:
        json.dump(test_questions, f, indent=2)

    return train_questions, val_questions, test_questions

def postprocess_vqa_data(all_vqa_questions, seg_id_list=(), max_num_of_seg_ids_per_empty_count=100,
                         default_modality="t1c", save_vqa_file="brats_gli_vqa_clean_data.json", seed=0):
    if seg_id_list:
        filtered_vqa_questions = [q for q in all_vqa_questions if q["seg_id"] in seg_id_list]
    elif max_num_of_seg_ids_per_empty_count is not None:
        filtered_vqa_questions = filter_seg_ids_from_vqa_data(all_vqa_questions,
                                                              max_num_of_seg_ids_per_empty_count=max_num_of_seg_ids_per_empty_count,
                                                              seed=seed)
    for index in range(len(filtered_vqa_questions)):
        question = filtered_vqa_questions[index]
        base_dir = os.path.basename(os.path.dirname(question["seg_file"]))
        question["img_id"] = filtered_vqa_questions[index]["seg_id"]
        if "img_name" not in question:
            base_img_file = os.path.basename(question["seg_file"]).replace("seg", default_modality)
            question["img_name"] = os.path.join(base_dir, base_img_file)
        assert "question" in filtered_vqa_questions[index]
        assert "answer" in filtered_vqa_questions[index]
        question["q_lang"] = "en"
        question["qid"] = index
        question["location"] = "Brain"
        if "modality" not in question:
            question["modality"] = default_modality
        question["answer_type"] = "OPEN"
        question["base_type"] = "VQA"
        question["content_type"] = question["type"]
        question["qid"] = index
        question["study_name"] = "-".join(base_dir.split("-")[:-1])

    with open(save_vqa_file, 'w') as f:
        json.dump(filtered_vqa_questions, f, indent=2)

    return filtered_vqa_questions


def filter_seg_ids_from_vqa_data(all_vqa_questions, max_num_of_seg_ids_per_empty_count=100, seed=0):
    seg_ids_empty_counts_map = get_seg_ids_empty_counts(all_vqa_questions)
    random_state = np.random.RandomState(seed)

    # 1) Group seg_ids by their empties count
    count_to_segids = defaultdict(list)
    for seg_id, empties_val in seg_ids_empty_counts_map.items():
        count_to_segids[empties_val].append(seg_id)

    # 2) For each empties_val group, if it has more than max_segids_per_count seg_ids,
    #    we randomly sample up to that limit. Otherwise, keep all.
    final_seg_ids = []
    for empties_val, segids in count_to_segids.items():
        if len(segids) > max_num_of_seg_ids_per_empty_count:
            chosen = random_state.choice(segids, size=max_num_of_seg_ids_per_empty_count, replace=False)
        else:
            chosen = segids
        final_seg_ids.extend(chosen)
    return [q for q in all_vqa_questions if q["seg_id"] in final_seg_ids]


def generate_modality_question(modality):
    question = f"What is the modality of the brain image?"
    answer = modality
    question_dict = {"question": question, "answer": answer, "type": "modality", "label_name": "NA"}
    return [question_dict]


def generate_labal_vqa_questions(summ, include_area=True, include_quadrant=True, include_bbox=True, include_extent=True,
                                 include_solidity=True, subjective_only=False):
    vqa_questions = []
    if include_area:
        question = f"How large is the area covered by {summ['name']}?"
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
        question = f"The smallest bounding box surrounding {summ['name']} is in which quadrants?"
        answer = f"{summ['bbox_str']}"
        question_dict = {"question": question, "answer": answer, "type": "bbox", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_extent:
        question = f"Within the smallest bounding box surrounding {summ['name']}, to what extent is the bounding box region filled?"
        if subjective_only:
            answer = f"{summ['extent_interp']}"
        else:
            answer = f"{summ['extent_value']:.1f}%, which is {summ['extent_interp']}"
        question_dict = {"question": question, "answer": answer, "type": "extent", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_solidity:
        question = f"Within the smallest bounding box surrounding {summ['name']}, how solid is the region?"
        if subjective_only:
            answer = f"{summ['solidity_interp']}"
        else:
            answer = f"{summ['solidity_value']:.1f}%, which is {summ['solidity_interp']}"
        question_dict = {"question": question, "answer": answer, "type": "solidity", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    return vqa_questions


def get_descriptive_statistics(list_of_scores, zero_score_count, none_score_count, metric_name):
    lines = []
    avg_score = sum(list_of_scores) / len(list_of_scores)
    q1_score = np.quantile(list_of_scores, 0.25)
    q2_score = np.quantile(list_of_scores, 0.5)
    q3_score = np.quantile(list_of_scores, 0.75)
    min_score = min(list_of_scores)
    max_score = max(list_of_scores)

    # add non-zero scores
    non_zero_scores = [p for p in list_of_scores if p != 0.0]
    non_zero_avg_score = sum(non_zero_scores) / len(non_zero_scores)
    non_zero_q1_score = np.quantile(non_zero_scores, 0.25)
    non_zero_q2_score = np.quantile(non_zero_scores, 0.5)
    non_zero_q3_score = np.quantile(non_zero_scores, 0.75)
    non_zero_min_score = min(non_zero_scores)
    non_zero_max_score = max(non_zero_scores)

    return (f"\n{metric_name} questions:\n"
            f"  Count: {len(list_of_scores)}\n"
            f"  Avg:   {avg_score:.2f}%\n"
            f"  25-50-75: [{q1_score:.2f}, {q2_score:.2f}, {q3_score:.2f}]\n"
            f"  Range: [{min_score:.2f}%, {max_score:.2f}%]\n"
            f"  # with 0% {metric_name}: {zero_score_count}\n"
            f"  # with none {metric_name}: {none_score_count}\n"

            f"\n (non-zero) {metric_name} questions:\n"
            f"  Count: {len(non_zero_scores)}\n"
            f"  Avg:   {non_zero_avg_score:.2f}%\n"
            f"  25-50-75: [{non_zero_q1_score:.2f}, {non_zero_q2_score:.2f}, {non_zero_q3_score:.2f}]\n"
            f"  Range: [{non_zero_min_score:.2f}%, {non_zero_max_score:.2f}%]\n")



def extract_label_intensity(
    image: np.ndarray,
    mask: np.ndarray,
    abs_intensity_diff_thresh: float
):
    """
    Given an MR image (e.g. T1CE) and a label mask, compute:
      1) the average intensity within the label region.
      2) whether this label's region has sufficiently high intensity
         compared to the surrounding area (based on difference_factor).

    :param image: 3D numpy array (or 2D if slice-based) of the MR modality
    :param mask:  3D (or 2D) binary segmentation mask for a particular label
    :param abs_intensity_diff_thresh: Thresh by which the label region must differ
                              from surrounding intensity
    :return: (label_is_present, avg_intensity, avg_surrounding_intensity)
             label_is_present is a boolean indicating if the label is considered present
             avg_intensity is the average intensity inside the mask
             avg_surrounding_intensity is the average intensity in the surrounding area
    """
    # 1) Extract average intensity in the mask region
    masked_intensities = image[mask > 0]
    avg_intensity = np.mean(masked_intensities)


    # 2) Compute the average intensity of the surrounding area
    #    - We perform a binary dilation on the mask to define an approximate neighborhood.
    #    - Then we take all voxels in the dilated region that are not in the original mask.
    structure = generate_binary_structure(rank=mask.ndim, connectivity=1)
    dilated_mask = binary_dilation(mask, structure=structure, iterations=2)  # e.g. 2 iterations
    surrounding_mask = np.logical_and(dilated_mask, np.logical_not(mask))

    surrounding_intensities = image[surrounding_mask > 0]
    if len(surrounding_intensities) == 0:
        return False, avg_intensity, 0.0

    avg_surrounding_intensity = np.mean(surrounding_intensities)

    # 3) Check if label region is significantly different from surroundings
    if np.abs(avg_intensity - avg_surrounding_intensity) < abs_intensity_diff_thresh:
        return False, avg_intensity, avg_surrounding_intensity

    # If all checks passed, we consider the label present
    return True, avg_intensity, avg_surrounding_intensity


def extract_label_intensity_components(
    image: np.ndarray,
    mask: np.ndarray,
    abs_intensity_diff_thresh: float,
    dilation_iterations: int = 2
):
    """
    Given an MR image (e.g. T1CE) and a segmentation mask (possibly with multiple
    connected components), this function:
      1) Identifies connected components in the mask.
      2) For each connected component, computes:
         - the average intensity within that component.
         - the average intensity in the local 'surrounding' region (via dilation).
         - checks if the absolute difference is above abs_intensity_diff_thresh.
      3) Returns a final mask that is the UNION of all connected components that
         pass the threshold criterion, as well as a boolean flag indicating if
         there is at least one component meeting the criterion.

    :param image: 3D numpy array (or 2D) of the MR modality
    :param mask:  3D (or 2D) binary segmentation mask
    :param abs_intensity_diff_thresh: Threshold by which each connected component
                                      must differ from its surroundings
    :param dilation_iterations: Number of iterations for the surrounding dilation
    :return: kept_mask, label_is_present, avg_intensities, avg_surroundings
             where:
                 - kept_mask is a binary mask containing only those components
                   whose difference from surrounding is >= threshold.
                 - label_is_present is True if any component meets the criterion.
                 - avg_intensities is a list of average intensities for each
                   connected component (in the order of component labels).
                 - avg_surroundings is a list of average surrounding intensities
                   for each connected component (same order).
    """

    # 1) Label the connected components in the mask
    labeled_mask, num_components = label_(mask)
    if num_components == 0:
        # No components at all
        return np.zeros_like(mask, dtype=bool), False, 0.0, 0.0

    # Prepare outputs
    kept_components = []
    avg_intensities = []
    avg_surroundings = []

    # Define the structure for dilation
    structure = generate_binary_structure(rank=mask.ndim, connectivity=1)

    # 2) Iterate over each connected component
    for comp_idx in range(1, num_components + 1):
        component_mask = (labeled_mask == comp_idx)

        # (a) Average intensity in this component
        component_intensities = image[component_mask]
        if component_intensities.size == 0:
            # Should not happen if label() found a component, but just in case
            avg_intensity = 0.0
        else:
            avg_intensity = component_intensities.mean()

        # (b) Average intensity of the surrounding region
        dilated_mask = binary_dilation(
            component_mask,
            structure=structure,
            iterations=dilation_iterations
        )
        surrounding_mask = np.logical_and(dilated_mask, np.logical_not(component_mask))

        surrounding_intensities = image[surrounding_mask]
        if surrounding_intensities.size == 0:
            # If there's no valid surrounding, skip or treat as failing threshold
            avg_surrounding_intensity = 0.0
        else:
            avg_surrounding_intensity = surrounding_intensities.mean()

        # Store these for reference
        avg_intensities.append(avg_intensity)
        avg_surroundings.append(avg_surrounding_intensity)

        # (c) Check if this component's difference meets threshold
        if abs(avg_intensity - avg_surrounding_intensity) >= abs_intensity_diff_thresh:
            kept_components.append(comp_idx)

    # 3) Construct the union mask of all kept connected components
    if len(kept_components) == 0:
        kept_mask = np.zeros_like(mask, dtype=bool)
        label_is_present = False
    else:
        # Create a binary mask that is True where the component label is in kept_components
        kept_mask = np.isin(labeled_mask, kept_components)
        label_is_present = True

    return kept_mask, label_is_present, np.mean(avg_intensities), np.mean(avg_surroundings)



def get_seg_ids_empty_counts(all_vqa_questions):
    empty_count_map = defaultdict(int)
    for q in all_vqa_questions:
        q_type = q.get("type", "unknown")
        answer = q.get("answer", "")
        seg_id = q.get("seg_id", "unknown")
        empty_count_map[seg_id]

        # Parse adjacency area questions
        if q_type == "adj_area":
            # Typical answer format: "25.0%, which is the majority"
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                adj_val = float(match.group(1))
                if adj_val == 0.0:
                    empty_count_map[seg_id] += 1

        # Parse adjacency quadrant questions
        if q_type == "adj_quadrants":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1

        # Parse area questions
        if q_type == "area":
            # Typical answer format: "25.0%, which is a small portion"
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                area_val = float(match.group(1))
                if area_val == 0.0:
                    empty_count_map[seg_id] += 1

        # Check bounding box questions
        if q_type == "bbox":
            # Typical answer format: "top-left, bottom-right" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "quadrant":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "extent":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "solidity":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1
    return empty_count_map


def analyze_label_relationship(maskA, maskB, total_pixels, height, width):
    """
    Analyze adjacency between maskB and maskA, reporting:
      - % of maskB that is adjacent to maskA
      - bounding box quadrants for adjacent vs. non-adjacent
      - subjective interpretations
    """
    adjacent_mask_B, non_adjacent_mask_B = compute_connected_component_adjacency_masks(maskA, maskB)
    adjacent_pct = compute_adj_percentage(adjacent_mask_B, maskB)
    non_adjacent_pct = 100 - adjacent_pct

    bbox_adjacent = compute_bounding_box(adjacent_mask_B, total_pixels)
    bbox_non_adjacent = compute_bounding_box(non_adjacent_mask_B, total_pixels)

    quadrants_adjacent = get_bounding_box_quadrants(bbox_adjacent, height, width)
    quadrants_non_adjacent = get_bounding_box_quadrants(bbox_non_adjacent, height, width)

    return {
        "adjacent_percentage": adjacent_pct,
        "non_adjacent_percentage": non_adjacent_pct,
        "adjacent_quadrants": quadrants_adjacent,
        "non_adjacent_quadrants": quadrants_non_adjacent,
        "adjacent_interpretation": interpret_relationship_percentage(adjacent_pct),
        "non_adjacent_interpretation": interpret_relationship_percentage(non_adjacent_pct)
    }


def get_label_mask(seg_map_2d, label):
    if label in [1, 2, 3, 4]:
        return seg_map_2d == label
    elif label == 5:
        return (seg_map_2d == 1) | (seg_map_2d == 3)
    else:
        raise ValueError(f"Invalid label: {label}")


def analyze_label_summary(seg_map_2d, height, width, total_pixels, image=None, abs_intensity_diff_thresh=10,
                          labels_order=(1, 2, 3, 4, 5), pediatric=False, goat=False):
    """
    For each label (1..4), compute:
      - area percentage + subjective interpretation
      - centroid quadrant
      - bounding box quadrants
      - extent-based "compactness" measure
    """
    label_summaries = []

    for lbl in labels_order:
        mask = get_label_mask(seg_map_2d, lbl)
        if image is not None:
            mask, _, _, _ = extract_label_intensity_components(image=image, mask=mask,
                                                               abs_intensity_diff_thresh=abs_intensity_diff_thresh)
        if goat:
            label_name = goat_label_names.get(lbl, f"Label {lbl}")
        elif pediatric:
            label_name = ped_label_names.get(lbl, f"Label {lbl}")
        else:
            label_name = label_names.get(lbl, f"Label {lbl}")

        area_pct = compute_area_percentage(mask, total_pixels)
        area_interp = interpret_area_percentage(area_pct)

        if area_interp == "none":
            centroid = None
            quadrant = "none"
            bounding_box_quads = None
            bounding_box_str = "none"
            extent_value = 0.0
            extent_interp = "none"
            solidity_value = 0.0
            solidity_interp = "none"
            shape_desc = {}
        else:
            centroid = center_of_mass(mask)
            quadrant = get_quadrant(centroid, height, width)

            bbox = compute_bounding_box(mask, total_pixels)
            bounding_box_quads = get_bounding_box_quadrants(bbox, height, width)
            bounding_box_str = bounding_box_quads if bounding_box_quads else "none"

            # Extent-based compactness
            extent_value, extent_interp = measure_extent_compactness(mask)
            solidity_value, solidity_interp = measure_solidity(mask)

        label_summaries.append({
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
            "solidity_interp": solidity_interp,
        })
    return label_summaries


def analyze_segmentation_map(seg_map_2d):
    """
    Master function to produce a textual report combining:
      - Label summaries (area %, quadrant, bounding box, extent-based compactness)
        with subjective interpretations.
      - Non-Enh vs Enh tumor adjacency info.
      - FLAIR vs Tumor Core adjacency info.
      - Resection cavity vs tumor core & FLAIR.
    """
    height, width = seg_map_2d.shape
    total_pixels = seg_map_2d.size

    # Summaries of labels
    label_summaries = analyze_label_summary(seg_map_2d, height, width, total_pixels)

    # Create masks for relationships
    mask1 = (seg_map_2d == 1)  # Non-Enh
    mask2 = (seg_map_2d == 2)  # FLAIR
    mask3 = (seg_map_2d == 3)  # Enh
    mask4 = (seg_map_2d == 4)  # Resection cavity
    tumor_core_mask = np.logical_or(mask1, mask3)  # Non-Enh + Enh

    # Relationship: Non-Enh (1) vs. Enh (3)
    rel_1_vs_3 = analyze_label_relationship(mask1, mask3, total_pixels, height, width)

    # Relationship: FLAIR (2) vs. Tumor Core (1+3)
    rel_2_vs_core = analyze_label_relationship(tumor_core_mask, mask2, total_pixels, height, width)

    # Relationship: Resection Cavity (4) vs Tumor Core
    rel_4_vs_core = analyze_label_relationship(tumor_core_mask, mask4, total_pixels, height, width)

    # Relationship: Resection Cavity (4) vs FLAIR (2)
    rel_4_vs_2 = analyze_label_relationship(mask2, mask4, total_pixels, height, width)

    #######################################################################
    # Build the text report
    #######################################################################
    lines = []

    lines.append("===== LABEL SUMMARIES =====")
    for summ in label_summaries:
        lines.append(f"{summ['name']} (Label {summ['label']}):")
        lines.append(f"  - Area covers ~{summ['area_pct']:.2f}% of total pixels, which is {summ['area_interp']}.")
        lines.append(f"  - Centered in the {summ['centroid_quadrant']} quadrant.")
        lines.append(f"  - Bounding box quadrants: {summ['bbox_str'] or 'none'}")
        lines.append(f"  - Within its bounding box, it is {summ['extent_interp']} (extent={summ['extent_value']:.2f}) and {summ['solidity_interp']} (solidity={summ['solidity_value']:.2f})")
        lines.append("")

    # 1) Non-Enh vs. Enh Tumor
    lines.append("===== NON-ENHANCING TUMOR (Label 1) vs. ENHANCING TUMOR (Label 3) =====")
    lines.append(f"Enhancing tumor adjacent to non-enhancing tumor: ~{rel_1_vs_3['adjacent_percentage']:.2f}% ({rel_1_vs_3['adjacent_interpretation']}).")
    lines.append(f"Enhancing tumor NOT adjacent: ~{rel_1_vs_3['non_adjacent_percentage']:.2f}% ({rel_1_vs_3['non_adjacent_interpretation']}).")
    lines.append(f"Quadrants of adjacent regions: {', '.join(rel_1_vs_3['adjacent_quadrants']) or 'none'}")
    lines.append(f"Quadrants of non-adjacent regions: {', '.join(rel_1_vs_3['non_adjacent_quadrants']) or 'none'}\n")

    # 2) FLAIR vs. Tumor Core
    lines.append("===== FLAIR HYPERINTENSITY (Label 2) vs. TUMOR CORE (Labels 1 & 3) =====")
    lines.append(f"FLAIR area adjacent to tumor core: ~{rel_2_vs_core['adjacent_percentage']:.2f}% ({rel_2_vs_core['adjacent_interpretation']}).")
    lines.append(f"FLAIR area NOT adjacent: ~{rel_2_vs_core['non_adjacent_percentage']:.2f}% ({rel_2_vs_core['non_adjacent_interpretation']}).")
    lines.append(f"Quadrants of adjacent regions: {', '.join(rel_2_vs_core['adjacent_quadrants']) or 'none'}")
    lines.append(f"Quadrants of non-adjacent regions: {', '.join(rel_2_vs_core['non_adjacent_quadrants']) or 'none'}\n")

    # 3) Resection Cavity (4) vs. Tumor Core
    lines.append("===== RESECTION CAVITY (Label 4) vs. TUMOR CORE (Labels 1 & 3) =====")
    lines.append(f"Resection cavity area adjacent to tumor core: ~{rel_4_vs_core['adjacent_percentage']:.2f}% ({rel_4_vs_core['adjacent_interpretation']}).")
    lines.append(f"Resection cavity area NOT adjacent: ~{rel_4_vs_core['non_adjacent_percentage']:.2f}% ({rel_4_vs_core['non_adjacent_interpretation']}).")
    lines.append(f"Quadrants of adjacent regions: {', '.join(rel_4_vs_core['adjacent_quadrants']) or 'none'}")
    lines.append(f"Quadrants of non-adjacent regions: {', '.join(rel_4_vs_core['non_adjacent_quadrants']) or 'none'}\n")

    # 4) Resection Cavity (4) vs. FLAIR (2)
    lines.append("===== RESECTION CAVITY (Label 4) vs. FLAIR (Label 2) =====")
    lines.append(f"Resection cavity area adjacent to FLAIR: ~{rel_4_vs_2['adjacent_percentage']:.2f}% ({rel_4_vs_2['adjacent_interpretation']}).")
    lines.append(f"Resection cavity area NOT adjacent: ~{rel_4_vs_2['non_adjacent_percentage']:.2f}% ({rel_4_vs_2['non_adjacent_interpretation']}).")
    lines.append(f"Quadrants of adjacent regions: {', '.join(rel_4_vs_2['adjacent_quadrants']) or 'none'}")
    lines.append(f"Quadrants of non-adjacent regions: {', '.join(rel_4_vs_2['non_adjacent_quadrants']) or 'none'}")

    return "\n".join(lines)


def vqa_round(value):
    """
    Round a value to a specific number of decimal places.
    """
    round_val = np.round(value, 2)
    if round_val >= 0.005 and round_val < 0.01:
        return 0.01
    if round_val < .005:
        return 0.0
    else:
        return round_val


def compute_bounding_box(mask, total_pixels):
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
    return min_r, min_c, max_r, max_c


def compute_area_percentage(mask, total_pixels):
    """
    Returns the percentage of 'mask' pixels relative to the total segmentation size.
    """
    return vqa_round((mask.sum() / total_pixels) * 100)


def compute_adj_percentage(adj_mask, orig_mask):
    """
    Returns the percentage of adject 'mask' pixels relative to the overall mask.
    """
    if orig_mask.sum() == 0:
        return 0
    else:
        return vqa_round((adj_mask.sum() / orig_mask.sum()) * 100)


def interpret_area_percentage(pct):
    """
    Subjective interpretation of area percentage, tuned for smaller values.
    Example thresholds (you can tweak these to your liking):
      - 0.0%:    "none"
      - <0.1%:   "almost negligible"
      - <0.5%:   "tiny fraction"
      - <2%:     "very small fraction"
      - <5%:     "small portion"
      - <10%:    "moderate portion"
      - <20%:    "significant portion"
      - <40%:    "large portion"
      - <70%:    "major portion"
      - >=70%:   "the vast majority"
    """
    if pct == 0.0:
        return "none"
    elif pct < 0.01:
        return "almost negligible"
    elif pct < 0.05:
        return "tiny fraction"
    elif pct < .1:
        return "very small fraction"
    elif pct < .3:
        return "small portion"
    elif pct < .6:
        return "moderate portion"
    elif pct < 1.2:
        return "significant portion"
    elif pct < 4.0:
        return "large portion"
    elif pct < 7.0:
        return "major portion"
    else:
        return "the vast majority"


def get_quadrant(centroid, height, width):
    """
    Maps a (row, col) centroid to one of 9 quadrants (top-left to bottom-right).
    """
    if not centroid or np.isnan(centroid[0]) or np.isnan(centroid[1]):
        return "none"

    row, col = centroid
    third_height = height / 3
    third_width = width / 3

    if row < third_height:  # top row
        if col < third_width:
            return "top-left"
        elif col < 2 * third_width:
            return "top-center"
        else:
            return "top-right"
    elif row < 2 * third_height:  # middle row
        if col < third_width:
            return "center-left"
        elif col < 2 * third_width:
            return "center-center"
        else:
            return "center-right"
    else:  # bottom row
        if col < third_width:
            return "bottom-left"
        elif col < 2 * third_width:
            return "bottom-center"
        else:
            return "bottom-right"


def get_bounding_box_quadrants(bbox, height, width):
    """
    Determine which quadrants are affected by a bounding box
    by sampling corners of the bounding box.
    """
    if not bbox:
        return "none"

    min_r, min_c, max_r, max_c = bbox
    affected_quadrants = set()

    # We check the four corners
    corners = [
        (min_r, min_c),
        (min_r, max_c - 1),
        (max_r - 1, min_c),
        (max_r - 1, max_c - 1),
    ]
    for (r, c) in corners:
        quadrant = get_quadrant((r, c), height, width)
        if quadrant != "none":
            affected_quadrants.add(quadrant)
    if len(affected_quadrants) == 0:
        return "none"
    affected_quadrants_str = ", ".join(sorted(affected_quadrants))
    return affected_quadrants_str


def compute_connected_component_adjacency_masks(maskA, maskB, dilation_iterations=1):
    """
    Identify which connected components of maskB are adjacent to maskA
    (within a specified dilation). Returns two binary masks:
      1) B_adj_mask   -> Union of all B’s connected components that touch A
      2) B_nonadj_mask-> Union of all B’s connected components that do not touch A

    Parameters
    ----------
    maskA : 2D bool array
        Binary mask for label A.
    maskB : 2D bool array
        Binary mask for label B.
    dilation_iterations : int, default=1
        Number of times to dilate A to consider adjacency.
        1 => one-pixel boundary expansion.

    Returns
    -------
    B_adj_mask : 2D bool array
        All connected components in B that are adjacent to A.
    B_nonadj_mask : 2D bool array
        All connected components in B that are not adjacent to A.
    labeledB : 2D int array
        Connected component labels for B (same shape as maskB),
        where 0 => background; 1..N => different connected components.
    """

    # 1) Dilate A so boundary contact is recognized
    dilatedA = maskA.copy()
    for _ in range(dilation_iterations):
        dilatedA = binary_dilation(dilatedA)

    # 2) Label each connected component in B
    labeledB = label(maskB, connectivity=2)  # 8-connected

    # Initialize empty masks for adjacency
    B_adj_mask = np.zeros_like(maskB, dtype=bool)
    B_nonadj_mask = np.zeros_like(maskB, dtype=bool)

    max_label = labeledB.max()
    for cc_id in range(1, max_label + 1):
        component_mask = (labeledB == cc_id)

        # Check overlap with dilated A
        overlap = np.logical_and(dilatedA, component_mask).any()

        if overlap:
            # Entire connected component belongs to the "adjacent" mask
            B_adj_mask |= component_mask
        else:
            # Entire connected component belongs to the "non-adjacent" mask
            B_nonadj_mask |= component_mask

    return B_adj_mask, B_nonadj_mask



###############################################################################
# Measuring Compactness via "Extent" (Region's fill within its bounding box)
###############################################################################

def measure_extent_compactness(mask, total_pixels):
    """
    Extent = region_area / bounding_box_area.
    Returns a float in [0..1], plus a subjective interpretation:
      - <0.2 => "very sparse"
      - 0.2..0.5 => "somewhat scattered"
      - 0.5..0.8 => "partially filling"
      - 0.8..0.95 => "nearly filling"
      - >0.95 => "almost fully filling"
    """
    bbox = compute_bounding_box(mask, total_pixels)
    area = mask.sum()
    if not bbox:
        return 0.0, "none"

    min_r, min_c, max_r, max_c = bbox
    bbox_h = max_r - min_r
    bbox_w = max_c - min_c

    bbox_area = bbox_h * bbox_w
    if bbox_area == 0:
        return 0.0, "none"

    extent = (area / bbox_area) * 100
    interpretation = interpret_extent(extent)
    return extent, interpretation


def interpret_extent(value):
    """
    Subjective interpretation of how well the region fills its bounding box.
    """
    if value == 0.0:
        return "none"
    elif value < 20.0:
        return "very sparse"
    elif value < 50.0:
        return "somewhat scattered"
    elif value < 80.0:
        return "partially filled"
    elif value < 95.0:
        return "nearly filled"
    else:
        return "almost fully filled"


def measure_solidity(mask):
    """
    Computes Solidity = area / convex_hull_area.
    """
    convex_hull = convex_hull_image(mask)
    region_area = mask.sum()
    convex_area = convex_hull.sum()
    solidity = vqa_round((region_area / convex_area) * 100) if convex_area > 0 else 0.0
    return solidity, interpret_solidity(solidity)


def interpret_solidity(value):
    """
    Subjective interpretation of solidity.
    """
    if value == 0.0:
        return "none"
    elif value < 60:
        return "highly irregular and scattered"
    elif value < 80.0:
        return "somewhat compact but irregular"
    else:
        return "mostly compact"


def interpret_relationship_percentage(pct):
    """
    Subjective interpretation of adjacency / overlap percentages:
     - <5% => "minimal"
     - 5-20% => "some"
     - 20-50% => "a moderate amount"
     - 50-80% => "the majority"
     - >80% => "the vast majority"
    """
    if pct == 0:
        return "none"
    if pct < 5:
        return "minimal"
    elif pct < 20:
        return "some"
    elif pct < 50:
        return "a moderate amount"
    elif pct < 80:
        return "the majority"
    else:
        return "the vast majority"

def summarize_vqa_data(all_vqa_questions,
                       max_seg_id_list=(10, 20, 25, 30, 40, 50, 70, 77, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)):
    """
    Summarize basic statistics about the generated VQA data:
      - Count total questions
      - Distribution of question types
      - Distribution of label names
      - Parse adjacency and area percentages for min/max/avg
      - Count occurrences where we have 'none' labels or 0% adjacency
    """
    total_questions = len(all_vqa_questions)
    question_type_counts = Counter(q.get("type", "unknown") for q in all_vqa_questions)
    label_name_counts = Counter(q.get("label_name", "unknown") for q in all_vqa_questions)

    # We'll try to parse numeric percentages for adjacency (adj_area) questions:
    adjacency_percentages = []
    # Keep track of how many adjacency questions are effectively 0% => "none" adjacency
    zero_adjacency_count = 0
    none_adj_quadrant_count = 0

    # We'll do the same for area questions:
    area_percentages = []
    # Keep track of how many times area=0% => "none" label
    zero_area_count = 0
    none_area_count = 0

    extent_percentages = []
    solidity_percentages = []

    # We'll also check how often bounding box = "none" or quadrant = "none"
    # in quadrant-related or bbox-related questions
    none_quadrant_count = 0
    none_bbox_count = 0
    none_extent_count = 0
    none_solidity_count = 0

    completely_empty_map = dict()
    empty_count_map = defaultdict(int)
    no_tumor_core_map = dict()

    for q in all_vqa_questions:
        q_type = q.get("type", "unknown")
        answer = q.get("answer", "")
        seg_id = q.get("seg_id", "unknown")
        label_name = q.get("label_name", "unknown")
        empty_count_map[seg_id]

        # Parse adjacency area questions
        if q_type == "adj_area":
            # Typical answer format: "25.0%, which is the majority"
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                adj_val = float(match.group(1))
                adjacency_percentages.append(adj_val)
                if adj_val == 0.0:
                    zero_adjacency_count += 1
                    empty_count_map[seg_id] += 1

        # Parse adjacency quadrant questions
        if q_type == "adj_quadrants":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_adj_quadrant_count += 1
                empty_count_map[seg_id] += 1

        # Parse area questions
        if q_type == "area":
            # Typical answer format: "25.0%, which is a small portion"
            match = re.search(r'([\d.]+)%,', answer)
            area_val = None
            if match:
                area_val = float(match.group(1))
                area_percentages.append(area_val)
                if area_val == 0.0:
                    zero_area_count += 1
                    empty_count_map[seg_id] += 1
                    completely_empty_map[seg_id] = completely_empty_map[seg_id] and True if seg_id in completely_empty_map else True
                    if "tumor core" in label_name.lower():
                        no_tumor_core_map[seg_id] = True
                else:
                    completely_empty_map[seg_id] = False
            if "none" in answer.lower():
                none_area_count += 1
            if "none" not in answer.lower() and area_val == 0.0:
                print("Anomaly: ", answer)

        # Check bounding box questions
        if q_type == "bbox":
            # Typical answer format: "top-left, bottom-right" or "none"
            if "none" in answer.lower():
                none_bbox_count += 1
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "quadrant":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_quadrant_count += 1
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "extent":
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                extent_val = float(match.group(1))
                extent_percentages.append(extent_val)
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_extent_count += 1
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "solidity":
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                solidity_val = float(match.group(1))
                solidity_percentages.append(solidity_val)
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_solidity_count += 1
                empty_count_map[seg_id] += 1

    # Build the summary report lines
    lines = []
    lines.append("===== VQA DATA SUMMARY =====")
    lines.append(f"Total questions: {total_questions}")

    lines.append("\nQuestion type distribution:")
    for t, c in question_type_counts.items():
        lines.append(f"  - {t}: {c}")

    lines.append("\nLabel name distribution:")
    for lbl, c in label_name_counts.items():
        lines.append(f"  - {lbl}: {c}")

    # Summaries of adjacency values
    if adjacency_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=adjacency_percentages,
                                                zero_score_count=zero_adjacency_count,
                                                none_score_count=none_adj_quadrant_count, metric_name="adjacency_area"))

    # Summaries of area values
    if area_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=area_percentages, zero_score_count=zero_area_count,
                                                none_score_count=none_area_count, metric_name="area"))

    if extent_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=extent_percentages, zero_score_count=np.nan,
                                                none_score_count=none_extent_count, metric_name="extent"))

    if solidity_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=solidity_percentages, zero_score_count=np.nan,
                                                none_score_count=none_solidity_count, metric_name="solidity"))

    # Summaries of "none" answers for quadrant and bounding box
    lines.append(f"\n# of questions that returned 'none' quadrant: {none_quadrant_count}")
    lines.append(f"# of questions that returned 'none' bounding box: {none_bbox_count}")
    lines.append(f"# of questions that returned 'none' extent: {none_extent_count}")
    lines.append(f"# of questions that returned 'none' solidity: {none_solidity_count}")

    lines.append(f"# of seg maps: {len(completely_empty_map)}, seg maps that are empty: {sum(list(completely_empty_map.values()))}, seg maps without tumor core: {sum(list(no_tumor_core_map.values()))}")

    if empty_count_map is not None:
        #for remove_empty_counts in [[], [33, 28], [33, 28, 26]]:
        #    for empty_count in remove_empty_counts:
        #        empty_count_map = {seg_id: count for seg_id, count in list(empty_count_map.items()) if count != empty_count}
        empty_count_map_ = empty_count_map.copy()
        for max_seg_id in max_seg_id_list:
            empty_count_tracker = dict()
            empty_count_map = dict()
            for seg_id, count in list(empty_count_map_.items()):
                if count not in empty_count_tracker:
                    empty_count_tracker[count] = 1
                    empty_count_map[seg_id] = count
                else:
                    if empty_count_tracker[count] < max_seg_id:
                        empty_count_tracker[count] += 1
                        empty_count_map[seg_id] = count
            empty_counts = list(empty_count_map.values())
            avg_empty_counts = sum(empty_counts) / len(empty_counts)
            q1_empty_counts = np.quantile(empty_counts, 0.25)
            q2_empty_counts = np.quantile(empty_counts, 0.5)
            q3_empty_counts = np.quantile(empty_counts, 0.75)
            min_empty_counts = min(empty_counts)
            max_empty_counts = max(empty_counts)
            empty_count_distribution = Counter(empty_counts)
            seg_id_counts = list(empty_count_distribution.values())
            avg_seg_id_counts = sum(seg_id_counts) / len(seg_id_counts)
            q1_seg_id_counts = np.quantile(seg_id_counts, 0.25)
            q2_seg_id_counts = np.quantile(seg_id_counts, 0.5)
            q3_seg_id_counts = np.quantile(seg_id_counts, 0.75)
            min_seg_id_counts = min(seg_id_counts)
            max_seg_id_counts = max(seg_id_counts)
            lines.append(
                f"\nEmpty counts per seg_id (over 4 image types with 1 modality question and 5 questions for 5 organs and 2 questions for 4 relationships. total=136):\n"
                #f"\nEmpty counts per seg_id (over 5 questions for 5 organs and 2 questions for 4 relationships. total=33):\n"
                #f"  Removing seg_ids with empty counts: " + ", ".join([str(k) for k in remove_empty_counts]) + "\n"
                f"  Max # of seg_ids: {max_seg_id}\n"
                f"  Seg_id Count: {len(empty_count_map)}\n"
                f"  Avg Empty Count:   {avg_empty_counts:.2f}\n"
                f"  Empty Count 25-50-75: [{q1_empty_counts:.2f}, {q2_empty_counts:.2f}, {q3_empty_counts:.2f}]\n"
                f"  Empty Count Range: [{min_empty_counts:.2f}, {max_empty_counts:.2f}]\n"
                f"  Empty Count Distribution: {[(k, empty_count_distribution[k]) for k in sorted(empty_count_distribution.keys())]}\n"
                f"  Avg Seg_id Count:   {avg_seg_id_counts:.2f}\n"
                f"  Seg_id 25-50-75: [{q1_seg_id_counts:.2f}, {q2_seg_id_counts:.2f}, {q3_seg_id_counts:.2f}]\n"
                f"  Seg_id Range: [{min_seg_id_counts:.2f}, {max_seg_id_counts:.2f}]\n"
            )
    return "\n".join(lines)


def summarize_3d_vqa_data(all_vqa_questions,
                       max_seg_id_list=(10, 20, 25, 30, 40, 50, 70, 77, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)):
    """
    Summarize basic statistics about the generated VQA data:
      - Count total questions
      - Distribution of question types
      - Distribution of label names
      - Parse adjacency and area percentages for min/max/avg
      - Count occurrences where we have 'none' labels or 0% adjacency
    """
    total_questions = len(all_vqa_questions)
    question_type_counts = Counter(q.get("type", "unknown") for q in all_vqa_questions)
    label_name_counts = Counter(q.get("label_name", "unknown") for q in all_vqa_questions)

    # We'll try to parse numeric percentages for adjacency (adj_area) questions:
    adjacency_percentages = []
    # Keep track of how many adjacency questions are effectively 0% => "none" adjacency
    zero_adjacency_count = 0
    none_adj_quadrant_count = 0

    # We'll do the same for area questions:
    area_percentages = []
    # Keep track of how many times area=0% => "none" label
    zero_area_count = 0
    none_area_count = 0

    extent_percentages = []
    solidity_percentages = []

    # We'll also check how often bounding box = "none" or quadrant = "none"
    # in quadrant-related or bbox-related questions
    none_quadrant_count = 0
    none_bbox_count = 0
    none_extent_count = 0
    none_solidity_count = 0

    completely_empty_map = dict()
    empty_count_map = defaultdict(int)
    no_tumor_core_map = dict()

    for q in all_vqa_questions:
        q_type = q.get("type", "unknown")
        answer = q.get("answer", "")
        seg_id = q.get("volume_file_id", "unknown")
        label_name = q.get("label_name", "unknown")
        empty_count_map[seg_id]

        # Parse adjacency area questions
        if q_type == "adj_area":
            # Typical answer format: "25.0%, which is the majority"
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                adj_val = float(match.group(1))
                adjacency_percentages.append(adj_val)
                if adj_val == 0.0:
                    zero_adjacency_count += 1
                    empty_count_map[seg_id] += 1

        # Parse adjacency quadrant questions
        if q_type == "adj_quadrants":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_adj_quadrant_count += 1
                empty_count_map[seg_id] += 1

        # Parse area questions
        if q_type == "area":
            # Typical answer format: "25.0%, which is a small portion"
            match = re.search(r'([\d.]+)%,', answer)
            area_val = None
            if match:
                area_val = float(match.group(1))
                area_percentages.append(area_val)
                if area_val == 0.0:
                    zero_area_count += 1
                    empty_count_map[seg_id] += 1
                    completely_empty_map[seg_id] = completely_empty_map[seg_id] and True if seg_id in completely_empty_map else True
                    if "tumor core" in label_name.lower():
                        no_tumor_core_map[seg_id] = True
                else:
                    completely_empty_map[seg_id] = False
            if "none" in answer.lower():
                none_area_count += 1
            if "none" not in answer.lower() and area_val == 0.0:
                print("Anomaly: ", answer)

        # Check bounding box questions
        if q_type == "bbox":
            # Typical answer format: "top-left, bottom-right" or "none"
            if "none" in answer.lower():
                none_bbox_count += 1
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "quadrant":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_quadrant_count += 1
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "extent":
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                extent_val = float(match.group(1))
                extent_percentages.append(extent_val)
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_extent_count += 1
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "solidity":
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                solidity_val = float(match.group(1))
                solidity_percentages.append(solidity_val)
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_solidity_count += 1
                empty_count_map[seg_id] += 1

    # Build the summary report lines
    lines = []
    lines.append("===== VQA DATA SUMMARY =====")
    lines.append(f"Total questions: {total_questions}")

    lines.append("\nQuestion type distribution:")
    for t, c in question_type_counts.items():
        lines.append(f"  - {t}: {c}")

    lines.append("\nLabel name distribution:")
    for lbl, c in label_name_counts.items():
        lines.append(f"  - {lbl}: {c}")

    # Summaries of adjacency values
    if adjacency_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=adjacency_percentages,
                                                zero_score_count=zero_adjacency_count,
                                                none_score_count=none_adj_quadrant_count, metric_name="adjacency_area"))

    # Summaries of area values
    if area_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=area_percentages, zero_score_count=zero_area_count,
                                                none_score_count=none_area_count, metric_name="area"))

    if extent_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=extent_percentages, zero_score_count=np.nan,
                                                none_score_count=none_extent_count, metric_name="extent"))

    if solidity_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=solidity_percentages, zero_score_count=np.nan,
                                                none_score_count=none_solidity_count, metric_name="solidity"))

    # Summaries of "none" answers for quadrant and bounding box
    lines.append(f"\n# of questions that returned 'none' quadrant: {none_quadrant_count}")
    lines.append(f"# of questions that returned 'none' bounding box: {none_bbox_count}")
    lines.append(f"# of questions that returned 'none' extent: {none_extent_count}")
    lines.append(f"# of questions that returned 'none' solidity: {none_solidity_count}")

    lines.append(f"# of seg maps: {len(completely_empty_map)}, seg maps that are empty: {sum(list(completely_empty_map.values()))}, seg maps without tumor core: {sum(list(no_tumor_core_map.values()))}")
    return "\n".join(lines)
