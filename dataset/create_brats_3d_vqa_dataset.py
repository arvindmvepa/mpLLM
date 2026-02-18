from glob import glob
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import json
from create_brats_imaging_dataset import (get_nifti_seg_file_from_dir, get_nifti_t1_native_from_dir, get_nifti_non_seg_file_from_dir, load_lab_map_from_nifti)
from vqa_utils import generate_train_val_test_splits
from vqa_3d_utils import analyze_3d_label_summary, generate_3d_labal_vqa_questions_v3, postprocess_3d_vqa_data, summarise_vqa_stats


def generate_vqa_from_seg_map(volume_file_dir, volume_id, include_area=True, include_regions=True, include_shape=True,
                              include_satellite=True, labels_order=(1, 2, 3, 4), pediatric=False, goat=False):
    """
    Master function to produce a textual report combining:
      - Label summaries (area %, quadrant, bounding box, extent-based compactness)
        with subjective interpretations.
      - Non-Enh vs Enh tumor adjacency info.
      - FLAIR vs Tumor Core adjacency info.
      - Resection cavity vs tumor core & FLAIR.
    """
    nii_seg_file = get_nifti_seg_file_from_dir(volume_file_dir)
    nib_seg_map_3d, seg_map_3d = load_lab_map_from_nifti(nii_seg_file)
    nib_t1n_3d = get_nifti_t1_native_from_dir(volume_file_dir)

    height, width, depth = seg_map_3d.shape
    total_pixels = seg_map_3d.size

    all_vqa_questions = []

    # Summaries of labels
    label_summaries = analyze_3d_label_summary(nib_seg_map_3d=nib_seg_map_3d, seg_map_3d=seg_map_3d,
                                               nib_t1n_3d=nib_t1n_3d, height=height, width=width, depth=depth,
                                               total_pixels=total_pixels, labels_order=labels_order,
                                               pediatric=pediatric, goat=goat)
    vqa_questions = []
    # get single label questions
    for summ in label_summaries:
        label_vqa_questions = generate_3d_labal_vqa_questions_v3(summ=summ, include_area=include_area,
                                                                 include_regions=include_regions,
                                                                 include_shape=include_shape,
                                                                 include_satellite=include_satellite)
        vqa_questions.extend(label_vqa_questions)
    non_seg_files_dict = get_nifti_non_seg_file_from_dir(volume_file_dir)
    for q in vqa_questions:
        q["volume_file_id"] = volume_id
        q["volume_file_dir"] = volume_file_dir
        q["volume_seg_file"] = nii_seg_file
        q["volume_non_seg_files"] = non_seg_files_dict
    all_vqa_questions.extend(vqa_questions)
    return all_vqa_questions


def generate_vqa_data_from_seg_file_joblib(
    volume_file_dirs,
    n_jobs=-1,
    include_area=True,
    include_regions=True,
    include_shape=True,
    include_satellite=True,
    labels_order=(1, 2, 3, 4),
    pediatric=False,
    goat=False
):
    """
    Parallelized version of generating VQA data from a list of seg_files,
    with a progress bar (tqdm_joblib).

    Parameters
    ----------
    volume_file_dirs : list of str
        Paths to volume file dirs (NIFTI).
    n_jobs : int, default=-1
        Number of parallel jobs. -1 => use all cores.
    include_area, include_quadrant, include_bbox, ...
        Configuration flags passed down to generate_vqa_from_seg_map.
    """
    all_vqa_questions = []

    generate_vqa_from_seg_map

    # Wrap Parallel execution with tqdm_joblib for the progress bar:
    with tqdm_joblib(desc="Processing segmentation files", total=len(volume_file_dirs)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(generate_vqa_from_seg_map)(
                volume_file_dir,
                volume_id,
                include_area,
                include_regions,
                include_shape,
                include_satellite,
                labels_order,
                pediatric,
                goat
            )
            for volume_id, volume_file_dir in enumerate(volume_file_dirs)
        )

    # Combine results
    for r in results:
        all_vqa_questions.extend(r)

    return all_vqa_questions


if __name__ == "__main__":
    # reference vqa files to line up seg_ids and train/val/test splits
    ref_train_vqa_file = None
    ref_val_vqa_file = None
    ref_test_vqa_file = None
    # rest of the parameters
    subjective_only = True

    vqa_file = "brats_{}_3d_vqa_subj{}_data_{}.json"
    clean_vqa_file = "brats_{}_3d_vqa_subj{}_clean_data_{}.json"
    train_file = "brats_{}_3d_vqa_subj{}_train_{}.json"
    val_file = "brats_{}_3d_vqa_subj{}_val_{}.json"
    test_file = "brats_{}_3d_vqa_subj{}_test_{}.json"
    seed = 0

    # GLI dataset settings
    dataset_type = "gli"
    version = f"updated_v2_seed0{seed}"
    volume_file_dirs = sorted(list(glob(f'./BraTS2024-BraTS-GLI/training_data1_v2/*')))
    labels_order = (1, 2, 3, 4)
    pediatric = False
    goat = False

    # MET dataset settings
    #dataset_type = "met"
    #version = f"updated_v3_seed{seed}"
    #volume_file_dirs = sorted(list(glob(f'./BraTS2024-BraTS-MET/MICCAI-BraTS2024-MET-Challenge-Training_overall/*')))
    #labels_order = (1, 2, 3)
    #pediatric = False
    #goat = False

    # GoAT dataset settings
    #dataset_type = "goat"
    #version = f"updated_v3_seed{seed}"
    #volume_file_dirs = sorted(list(glob(f'./BraTS2024-BraTS-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/*')))
    #labels_order = (1, 2, 3)
    #pediatric = False
    #goat = True

    vqa_file = vqa_file.format(dataset_type, subjective_only, version)
    clean_vqa_file = clean_vqa_file.format(dataset_type, subjective_only, version)
    train_file = train_file.format(dataset_type, subjective_only, version)
    val_file = val_file.format(dataset_type, subjective_only, version)
    test_file = test_file.format(dataset_type, subjective_only, version)

    vqa_data_ = generate_vqa_data_from_seg_file_joblib(volume_file_dirs, labels_order=labels_order, n_jobs=8,
                                                       pediatric=pediatric, goat=goat)
    with open(vqa_file, 'w') as f:
        json.dump(vqa_data_, f, indent=2)

    with open(vqa_file, 'r') as f:
        vqa_data_ = json.load(f)
    stats = summarise_vqa_stats(vqa_data_)
    print("Total:", stats["total_questions"])
    for k, v in stats.items():
        if k == "total_questions":
            continue
        elif k == "answer_dist_per_type" or k == "answer_dist_per_label_and_type":
            for k_,v_ in v.items():
                v_.to_csv(f"{k}_{k_}.csv")
        else:
            v.to_csv(f"{k}.csv")
    processed_vqa_data = postprocess_3d_vqa_data(vqa_data_, save_vqa_file=clean_vqa_file)
    question_key = "volume_file_id"
    if (ref_train_vqa_file is not None) and (ref_val_vqa_file is not None) and (ref_test_vqa_file is not None):
        with open(ref_train_vqa_file, 'r') as f:
            ref_train_vqa_data = json.load(f)
            ref_train_seg_ids = [q[question_key] for q in ref_train_vqa_data]
        with open(ref_val_vqa_file, 'r') as f:
            ref_val_vqa_data = json.load(f)
            ref_val_seg_ids = [q[question_key] for q in ref_val_vqa_data]
        with open(ref_test_vqa_file, 'r') as f:
            ref_test_vqa_data = json.load(f)
            ref_test_seg_ids = [q[question_key] for q in ref_test_vqa_data]
        generate_train_val_test_splits(processed_vqa_data, question_key=question_key, train_seg_ids=ref_train_seg_ids,
                                       val_seg_ids=ref_val_seg_ids, test_seg_ids=ref_test_seg_ids,
                                       train_file=train_file, val_file=val_file, test_file=test_file)
    else:
        generate_train_val_test_splits(processed_vqa_data, question_key=question_key, train_file=train_file,
                                       val_file=val_file, test_file=test_file, seed=seed)
