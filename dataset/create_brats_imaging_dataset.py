import os
from glob import glob
import torch
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

# Example label-to-color map for segmentation masks
# Adjust these mappings if you have additional/different labels.
LABEL_COLORS = {
    0: (0, 0, 0),         # background: black
    1: (255, 0, 0),       # label 1: red
    2: (0, 255, 0),       # label 2: green
    3: (255, 255, 255),   # label 3: white
    4: (0, 0, 255),       # label 4: blue
}

COLOR_TO_LABEL = {v: k for k, v in LABEL_COLORS.items()}


def load_color_seg_png_as_labels(png_path):
    """
    Loads a color-coded PNG (RGB) where each label has a unique color,
    converts each pixel to integer labels, and returns a 2D NumPy array.
    """
    pil_img = Image.open(png_path).convert("RGB")
    rgb_data = np.array(pil_img)  # shape (H, W, 3)
    height, width = rgb_data.shape[:2]

    label_map = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            pixel_color = tuple(rgb_data[y, x])
            label_map[y, x] = COLOR_TO_LABEL.get(pixel_color, 0)  # default to 0 if not found
    return label_map


################################################################################
# 2) Convert (R,G,B) -> single integer => label
################################################################################
COLORINT_TO_LABEL = {
    ((r << 16) | (g << 8) | b): lbl
    for (r, g, b), lbl in COLOR_TO_LABEL.items()
}


def get_nifti_seg_file_from_dir(nii_file_dir: str) -> str:
    nii_files = glob(os.path.join(nii_file_dir, "*.nii.gz"))
    seg_nii_file = [nii_file for nii_file in nii_files if "seg" in nii_file][0]
    return seg_nii_file


def get_nifti_t1_native_from_dir(nii_file_dir: str) -> str:
    nii_files = glob(os.path.join(nii_file_dir, "*.nii.gz"))
    t1n_nii_file = [nii_file for nii_file in nii_files if "t1n" in nii_file][0]
    img = nib.load(t1n_nii_file)
    return img


def get_nifti_non_seg_file_from_dir(nii_file_dir: str) -> str:
    nii_dict = {}
    nii_files = glob(os.path.join(nii_file_dir, "*.nii.gz"))
    for modality in ["t1c", "t1n", "t2w", "t2f"]:
        nii_dict[modality] = [nii_file for nii_file in nii_files if modality+".nii.gz" in nii_file][0]
    return nii_dict


def load_lab_map_from_nifti(seg_nii_file: str) -> torch.Tensor:
    img = nib.load(seg_nii_file)
    label_map = img.get_fdata() # H x W x D
    return img, label_map


def load_color_seg_png_as_labels_gpu(png_path: str) -> torch.Tensor:
    """
    Reads a color-coded segmentation PNG on CPU, then does color->label
    mapping on the GPU. Returns a 2D Tensor of shape (H, W) with integer labels.
    """
    # ---------------------------
    # A) Load the image on CPU
    # ---------------------------
    pil_img = Image.open(png_path).convert("RGB")
    rgb_array = np.array(pil_img, dtype=np.uint8)  # shape: (H, W, 3)

    # ---------------------------
    # B) Move it to a GPU tensor
    # ---------------------------
    #   shape => (H, W, 3), dtype => uint8 -> int for bitwise ops
    rgb_tensor_cpu = torch.from_numpy(rgb_array)  # still on CPU
    rgb_tensor = rgb_tensor_cpu.to(device='cuda', non_blocking=True)

    # Flatten from (H, W, 3) to (H*W, 3)
    H, W, _ = rgb_tensor.shape
    flat_rgb = rgb_tensor.view(-1, 3).int()  # cast to int32 for bit shifts

    # ---------------------------
    # C) Compute single-integer color codes on GPU:
    #    color_int = (R<<16) | (G<<8) | B
    # ---------------------------
    R = flat_rgb[:, 0] << 16
    G = flat_rgb[:, 1] << 8
    B = flat_rgb[:, 2]
    color_int_tensor = R | G | B  # shape: (H*W,)

    # ---------------------------
    # D) Use Torch to find unique colors -> map each to label
    #    (similar to a dictionary lookup but vectorized)
    # ---------------------------
    unique_colors, inv_idx = torch.unique(color_int_tensor, return_inverse=True)
    # unique_colors: shape (U,) all distinct color ints in the image
    # inv_idx:       shape (H*W,) for each pixel, index into unique_colors

    # Convert our Python dict into torch Tensors on GPU
    dict_keys = list(COLORINT_TO_LABEL.keys())   # e.g., [0x000000, 0xFF0000, ...]
    dict_vals = list(COLORINT_TO_LABEL.values()) # e.g., [0, 1, ...]

    # Sort them for `searchsorted`
    dict_keys_t = torch.tensor(dict_keys, dtype=torch.int32, device='cuda')
    dict_vals_t = torch.tensor(dict_vals, dtype=torch.int32, device='cuda')
    sorted_keys, sort_idx = torch.sort(dict_keys_t)
    sorted_vals = dict_vals_t[sort_idx]

    # Sort unique_colors as well
    sorted_uc, sorted_uc_idx = torch.sort(unique_colors)

    # searchsorted -> indices in sorted_keys for each color in sorted_uc
    idx_in_dict = torch.searchsorted(sorted_keys, sorted_uc)
    # Clip to valid range
    idx_in_dict_clamped = torch.clamp(idx_in_dict, 0, sorted_keys.size(0) - 1)

    # Check if the color actually matches
    matched_mask = (sorted_uc == sorted_keys[idx_in_dict_clamped])
    matched_labels = torch.where(
        matched_mask,
        sorted_vals[idx_in_dict_clamped],
        torch.zeros_like(idx_in_dict_clamped, dtype=sorted_vals.dtype)
    )

    # Unsort so that matched_labels lines up with unique_colors
    inv_sort_idx = torch.argsort(sorted_uc_idx)
    final_labels_for_uc = matched_labels[inv_sort_idx]

    # Map back each pixel's color -> label
    label_map_flat = final_labels_for_uc[inv_idx].to(torch.uint8)  # shape (H*W,)

    # ---------------------------
    # E) Reshape -> (H, W) label map
    # ---------------------------
    label_map = label_map_flat.view(H, W)

    return label_map


def rank_slices_by_annotation(
    seg_dir: str,
    include_idxs=None
):
    """
    Ranks the y-slice indices by the total number of annotated pixels
    (summed across all volumes) in the given segmentation directory.

    :param seg_dir: Directory containing "*seg*.png" slices
                    (e.g., "Case001_seg_slice_9_y.png").
    :param include_idxs: Only consider these slice indices.
    :return: List of tuples (slice_idx, total_annotated_pixels)
             sorted in descending order.
    """
    slice_sums = {}

    # Gather PNG files from subdirectories that match "*seg*.png"
    seg_files = sorted(glob(os.path.join(seg_dir, "*", "*seg*.png")))
    if not seg_files:
        print(f"[INFO] No PNG files found in {seg_dir}.")
        return []

    for seg_file in tqdm(seg_files, desc="Processing slices on GPU"):
        # Example: "Case001_seg_slice_9_y.png"
        basename = os.path.basename(seg_file)
        parts = basename.split('_')
        # e.g. ["Case001", "seg", "slice", "9", "y.png"]
        slice_idx = int(parts[3])

        if include_idxs is not None and slice_idx not in include_idxs:
            continue

        # 1) Load & map to labels on GPU
        label_map = load_color_seg_png_as_labels_gpu(seg_file)

        # 2) Count non-zero pixels on GPU
        annotated_pixels_gpu = (label_map != 0).sum()

        # 3) Move count to CPU as an integer
        annotated_pixels = annotated_pixels_gpu.item()

        # 4) Accumulate
        slice_sums[slice_idx] = slice_sums.get(slice_idx, 0) + annotated_pixels

    # Sort slices by total annotated pixels in descending order
    ranked_slices = sorted(slice_sums.items(), key=lambda x: x[1], reverse=True)
    return ranked_slices


def convert_nifti_to_png_y_slices(
    input_root="/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2",
    output_root="output_pngs"
):
    """
    1) Recursively searches the 'input_root' directory for subject folders of the form 'BraTS-GLI-XXXXX-YYY'.
    2) Loads each .nii.gz file in the folder.
       - If it's a modality (t1n, t2w, etc.), saves grayscale slices.
       - If it's a segmentation (*seg.nii.gz), saves color-coded slices.
    3) Iterates over the Y dimension (data[:, y, :]) and saves each slice as PNG.
    4) Output filename format:
         <subjectID>_<modality>_slice_<y_index>_y.png
       e.g. BraTS-GLI-02208-100_t1n_slice_33_y.png (modality)
            BraTS-GLI-02208-100_seg_slice_33_y.png (segmentation)
    5) Output is placed into 'output_root/<subjectID>/' for tidy organization.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_root, exist_ok=True)

    # Find subject subdirectories
    subdirs = [
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d)) and d.startswith("BraTS-GLI-")
    ]

    for subdir in tqdm(subdirs):
        subdir_path = os.path.join(input_root, subdir)
        # Find all .nii.gz files
        nii_files = glob(os.path.join(subdir_path, "*.nii.gz"))

        # Create an output subfolder for this subject
        subject_out_dir = os.path.join(output_root, subdir)
        os.makedirs(subject_out_dir, exist_ok=True)

        for nii_file in nii_files:
            filename = os.path.basename(nii_file)
            # e.g. BraTS-GLI-03064-100-t1n.nii.gz -> "BraTS-GLI-03064-100-t1n"
            core_name = filename.replace(".nii.gz", "")
            # Typically: "BraTS-GLI-03064-100-t1n" => parts = ["BraTS", "GLI", "03064", "100", "t1n"]
            parts = core_name.split("-")
            subject_id = "-".join(parts[:4])  # BraTS-GLI-03064-100
            modality = parts[-1]             # e.g. "t1n", "t2w", "seg", etc.

            # Load the NIfTI volume
            img = nib.load(nii_file)
            data = img.get_fdata()  # shape often (X, Y, Z)

            # We'll generate coronal slices => data[:, y, :]
            num_slices = data.shape[1]  # Y dimension size

            # Decide if this file is a segmentation vs. a modality
            is_segmentation = "seg" in modality.lower()

            for y_index in range(num_slices):
                slice_data = data[:, y_index, :]  # shape (X, Z)

                if is_segmentation:
                    # Convert segmentation slice to color-coded image
                    slice_labels = slice_data.astype(np.uint8)  # integer labels
                    color_image = np.zeros((slice_labels.shape[0],
                                            slice_labels.shape[1], 3), dtype=np.uint8)

                    unique_labels = np.unique(slice_labels)
                    for label_val in unique_labels:
                        # Lookup color or default to white
                        color = LABEL_COLORS[label_val]
                        color_image[slice_labels == label_val] = color

                    pil_image = Image.fromarray(color_image, mode="RGB")

                else:
                    # Modality => grayscale
                    max_val = slice_data.max()
                    if max_val > 0:
                        slice_data = slice_data / max_val
                    slice_data = (slice_data * 255).astype(np.uint8)

                    # Convert to PIL image (could transpose if desired)
                    pil_image = Image.fromarray(slice_data)

                # Build output filename
                out_name = f"{subject_id}_{modality}_slice_{y_index}_y.png"
                out_path = os.path.join(subject_out_dir, out_name)
                pil_image.save(out_path)

                # Optional: print(f"Saved {out_path}")


if __name__ == "__main__":
    #convert_nifti_to_png_y_slices()
    segmentation_directory = "./output_pngs"

    # Get slice ranking
    ranking = rank_slices_by_annotation(
        seg_dir=segmentation_directory
    )

    if ranking:
        print("Slice Index Ranking by Annotated Pixels:")
        for rank_idx, (slice_idx, total_pixels) in enumerate(ranking, start=1):
            print(f"Rank {rank_idx}: Slice {slice_idx} => {total_pixels} annotated pixels")
    else:
        print("[INFO] No slices found or no annotations found.")
