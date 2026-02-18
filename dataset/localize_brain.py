from nilearn import plotting
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img, new_img_like
from nilearn.datasets import fetch_atlas_aal
import glob
import os
from tqdm import tqdm


# ---- 1. AAL names and indices  -------------------
AAL_NAMES = [
    'Precentral_L', 'Precentral_R',
    'Frontal_Sup_L', 'Frontal_Sup_R',
    'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R',
    'Frontal_Mid_L', 'Frontal_Mid_R',
    'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',
    'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R',
    'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',
    'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',
    'Rolandic_Oper_L', 'Rolandic_Oper_R',
    'Supp_Motor_Area_L', 'Supp_Motor_Area_R',
    'Olfactory_L', 'Olfactory_R',
    'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
    'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R',
    'Rectus_L', 'Rectus_R',
    'Insula_L', 'Insula_R',
    'Cingulum_Ant_L', 'Cingulum_Ant_R',
    'Cingulum_Mid_L', 'Cingulum_Mid_R',
    'Cingulum_Post_L', 'Cingulum_Post_R',
    'Hippocampus_L', 'Hippocampus_R',
    'ParaHippocampal_L', 'ParaHippocampal_R',
    'Amygdala_L', 'Amygdala_R',
    'Calcarine_L', 'Calcarine_R',
    'Cuneus_L', 'Cuneus_R',
    'Lingual_L', 'Lingual_R',
    'Occipital_Sup_L', 'Occipital_Sup_R',
    'Occipital_Mid_L', 'Occipital_Mid_R',
    'Occipital_Inf_L', 'Occipital_Inf_R',
    'Fusiform_L', 'Fusiform_R',
    'Postcentral_L', 'Postcentral_R',
    'Parietal_Sup_L', 'Parietal_Sup_R',
    'Parietal_Inf_L', 'Parietal_Inf_R',
    'SupraMarginal_L', 'SupraMarginal_R',
    'Angular_L', 'Angular_R',
    'Precuneus_L', 'Precuneus_R',
    'Paracentral_Lobule_L', 'Paracentral_Lobule_R',
    'Caudate_L', 'Caudate_R',
    'Putamen_L', 'Putamen_R',
    'Pallidum_L', 'Pallidum_R',
    'Thalamus_L', 'Thalamus_R',
    'Heschl_L', 'Heschl_R',
    'Temporal_Sup_L', 'Temporal_Sup_R',
    'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R',
    'Temporal_Mid_L', 'Temporal_Mid_R',
    'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R',
    'Temporal_Inf_L', 'Temporal_Inf_R',
    'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R',
    'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R',
    'Cerebelum_3_L', 'Cerebelum_3_R',
    'Cerebelum_4_5_L', 'Cerebelum_4_5_R',
    'Cerebelum_6_L', 'Cerebelum_6_R',
    'Cerebelum_7b_L', 'Cerebelum_7b_R',
    'Cerebelum_8_L', 'Cerebelum_8_R',
    'Cerebelum_9_L', 'Cerebelum_9_R',
    'Cerebelum_10_L', 'Cerebelum_10_R',
    'Vermis_1_2', 'Vermis_3', 'Vermis_4_5',
    'Vermis_6', 'Vermis_7', 'Vermis_8',
    'Vermis_9', 'Vermis_10',
]

AAL_INDICES = [
    '2001', '2002', '2101', '2102', '2111', '2112', '2201', '2202', '2211', '2212',
    '2301', '2302', '2311', '2312', '2321', '2322', '2331', '2332', '2401', '2402',
    '2501', '2502', '2601', '2602', '2611', '2612', '2701', '2702', '3001', '3002',
    '4001', '4002', '4011', '4012', '4021', '4022', '4101', '4102', '4111', '4112',
    '4201', '4202', '5001', '5002', '5011', '5012', '5021', '5022', '5101', '5102',
    '5201', '5202', '5301', '5302', '5401', '5402', '6001', '6002', '6101', '6102',
    '6201', '6202', '6211', '6212', '6221', '6222', '6301', '6302', '6401', '6402',
    '7001', '7002', '7011', '7012', '7021', '7022', '7101', '7102', '8101', '8102',
    '8111', '8112', '8121', '8122', '8201', '8202', '8211', '8212', '8301', '8302',
    '9001', '9002', '9011', '9012', '9021', '9022', '9031', '9032', '9041', '9042',
    '9051', '9052', '9061', '9062', '9071', '9072', '9081', '9082', '9100', '9110',
    '9120', '9130', '9140', '9150', '9160', '9170',
]


# ---- Define lobe membership by region name ---------------------

_FRONTAL = {
    'Precentral_L', 'Precentral_R',
    'Frontal_Sup_L', 'Frontal_Sup_R',
    'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R',
    'Frontal_Mid_L', 'Frontal_Mid_R',
    'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',
    'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R',
    'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',
    'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',
    'Rolandic_Oper_L', 'Rolandic_Oper_R',
    'Supp_Motor_Area_L', 'Supp_Motor_Area_R',
    'Olfactory_L', 'Olfactory_R',
    'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
    'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R',
    'Rectus_L', 'Rectus_R',
}

_INSULA = {'Insula_L', 'Insula_R'}

_LIMBIC = {
    'Cingulum_Ant_L', 'Cingulum_Ant_R',
    'Cingulum_Mid_L', 'Cingulum_Mid_R',
    'Cingulum_Post_L', 'Cingulum_Post_R',
    'Hippocampus_L', 'Hippocampus_R',
    'ParaHippocampal_L', 'ParaHippocampal_R',
    'Amygdala_L', 'Amygdala_R',
}

_OCCIPITAL = {
    'Calcarine_L', 'Calcarine_R',
    'Cuneus_L', 'Cuneus_R',
    'Lingual_L', 'Lingual_R',
    'Occipital_Sup_L', 'Occipital_Sup_R',
    'Occipital_Mid_L', 'Occipital_Mid_R',
    'Occipital_Inf_L', 'Occipital_Inf_R',
}

_TEMPORAL = {
    'Fusiform_L', 'Fusiform_R',
    'Heschl_L', 'Heschl_R',
    'Temporal_Sup_L', 'Temporal_Sup_R',
    'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R',
    'Temporal_Mid_L', 'Temporal_Mid_R',
    'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R',
    'Temporal_Inf_L', 'Temporal_Inf_R',
}

_PARietal = {
    'Postcentral_L', 'Postcentral_R',
    'Parietal_Sup_L', 'Parietal_Sup_R',
    'Parietal_Inf_L', 'Parietal_Inf_R',
    'SupraMarginal_L', 'SupraMarginal_R',
    'Angular_L', 'Angular_R',
    'Precuneus_L', 'Precuneus_R',
    'Paracentral_Lobule_L', 'Paracentral_Lobule_R',
}

_SUBCORTICAL = {
    'Caudate_L', 'Caudate_R',
    'Putamen_L', 'Putamen_R',
    'Pallidum_L', 'Pallidum_R',
    'Thalamus_L', 'Thalamus_R',
}

_CEREBELLUM = {
    'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R',
    'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R',
    'Cerebelum_3_L', 'Cerebelum_3_R',
    'Cerebelum_4_5_L', 'Cerebelum_4_5_R',
    'Cerebelum_6_L', 'Cerebelum_6_R',
    'Cerebelum_7b_L', 'Cerebelum_7b_R',
    'Cerebelum_8_L', 'Cerebelum_8_R',
    'Cerebelum_9_L', 'Cerebelum_9_R',
    'Cerebelum_10_L', 'Cerebelum_10_R',
    'Vermis_1_2', 'Vermis_3', 'Vermis_4_5',
    'Vermis_6', 'Vermis_7', 'Vermis_8',
    'Vermis_9', 'Vermis_10',
}

def _lobe_for_name(name: str) -> str:
    if name in _FRONTAL:
        return "frontal"
    if name in _PARietal:
        return "parietal"
    if name in _OCCIPITAL:
        return "occipital"
    if name in _TEMPORAL:
        return "temporal"
    if name in _LIMBIC:
        return "limbic"
    if name in _INSULA:
        return "insula"
    if name in _SUBCORTICAL:
        return "subcortical"
    if name in _CEREBELLUM:
        return "cerebellum"
    return "unknown"
    # (no brainstem regions in this particular list)


# ---- 3. Final index → lobe map ------------------------------------
AAL_INDEX_TO_LOBE: dict[int, str] = {
    int(idx): _lobe_for_name(name)
    for name, idx in zip(AAL_NAMES, AAL_INDICES)
}


# add background explicitly
AAL_INDEX_TO_LOBE[0] = "background"


def load_aal_atlas_label_map(version: str = "SPM12"):
    """
    Load AAL atlas from nilearn and build a mapping from atlas value -> label or lobe.

    For AAL, the integer values in the atlas image are in `indices`,
    and the human-readable names are in `labels`. We must use `indices`
    rather than assuming 1..N.  :contentReference[oaicite:0]{index=0}
    """
    aal = fetch_atlas_aal(version=version)
    atlas_img = nib.load(aal.maps)
    atlas_label_map = AAL_INDEX_TO_LOBE

    return atlas_img, atlas_label_map


def localize_to_brain_regions(
    tumour_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    atlas_label_map: dict[int, str],
    label_index: int = 1,
    debug=False,
    seg_path=None
) -> dict:
    """
    Parameters
    ----------
    tumour_img : nibabel image in patient space (binary or multi‑label seg)
    atlas_img  : nibabel image (anatomical atlas, same space)
    atlas_label_map : {int: str} mapping from atlas label index → region name
    tumour_label_value : which value inside tumour_img is the lesion mask
                         (e.g. 3 = enhancing, 2 = edema …)

    Returns
    -------
    results : dict
        {
          'total_voxels': int,
          'overlap': {
              atlas_index: {'region': str,
                            'voxels': int,
                            'percent': float}
              ...
          }
        }
    """
    # --- 1. resample atlas to tumour space if needed ---------------
    # (assumes atlas has been registered to the same physical space;
    #  we already reoriented both to canonical on load)
    if atlas_img.shape != tumour_img.shape or not np.allclose(atlas_img.affine, tumour_img.affine):
        atlas_img = resample_to_img(atlas_img, tumour_img, interpolation="nearest")

    # ---- drop trailing singleton dim if present --------------
    if atlas_img.ndim == 4 and atlas_img.shape[-1] == 1:
        atlas_img = new_img_like(atlas_img,
                                 atlas_img.get_fdata()[..., 0],  # squeeze
                                 atlas_img.affine)

    # --- 3. compute overlap ---------------------------------------
    tumour_mask = (tumour_img.get_fdata() == label_index)
    atlas_data = atlas_img.get_fdata().astype(np.int16)

    if debug:
        display = plotting.plot_roi(tumour_img,
                                    bg_img=atlas_img,
                                    title=f"Tumour-Affine Alignment Check Label {label_index}", alpha=0.5)
        display.savefig(f"tumour_affine_alignment_check_{os.path.basename(seg_path)}_label{label_index}.png")
        display.close()

    overlapped = atlas_data[tumour_mask]
    nonzero = overlapped[overlapped > 0]
    unique, counts = np.unique(nonzero, return_counts=True)
    total = int(tumour_mask.sum())
    overlap_voxels = int(nonzero.size)
    overlap_fraction = float(overlap_voxels) / float(total) if total else 0.0

    # --- 4. pack results ------------------------------------------
    overlap_dict = {}
    region_list = []
    for idx, cnt in zip(unique, counts):
        region = atlas_label_map.get(int(idx), "unknown")
        if region != "unknown":
            overlap_dict[int(idx)] = {
                "region": region,
                "voxels": int(cnt),
                "percent": float(cnt) * 100.0 / total if total else 0.0,
            }
            region_list.append(region)

    return {
        "total_voxels": total,
        "overlap_voxels": overlap_voxels,
        "overlap_fraction": overlap_fraction,
        "overlap": overlap_dict,
        "regions": sorted(set(region_list)),
    }


def get_region_str(region_list):
    """
    Helper function to convert the list of regions into a string
    """
    if len(region_list) == 0:
        return "N/A"
    elif len(region_list) == 1:
        return region_list[0]
    elif len(region_list) == 2:
        return f"{region_list[0]} and {region_list[1]}"
    else:
        # For more than two regions, join them with commas and 'and'
        return ", ".join(region_list[:-1]) + " and " + region_list[-1]


def analyze_label_localization(seg_path="/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/BraTS-GLI-00005-100/BraTS-GLI-00005-100-seg.nii.gz",
                               aal_version="SPM12", tumour_labels=None, debug=True):
    """
    seg_path      : path to your multi‑label tumour segmentation (NIfTI)
    atlas_path    : path to LPBA40 (or other) atlas NIfTI
    label_txt     : path to text file mapping atlas indices → region names
    tumour_labels : dict like {'ET': 3, 'SNFH': 2, 'NETC': 1, 'RC': 4}
                    (keys = your internal label names, values = voxel values)
    Returns
    -------
    summary : dict keyed by your tumour label
              e.g. summary['ET']['overlap'][46]['region'] → 'left‑MFG'
    """
    tumour_img = nib.as_closest_canonical(nib.load(seg_path))
    atlas_img, atlas_label_map = load_aal_atlas_label_map(version=aal_version)
    atlas_img = nib.as_closest_canonical(atlas_img)

    summary = {}
    for name, label_index in tumour_labels.items():
        summary[name] = localize_to_brain_regions(tumour_img=tumour_img, atlas_img=atlas_img,
                                                  atlas_label_map=atlas_label_map,
                                                  label_index=label_index, debug=debug, 
                                                  seg_path=seg_path)

    return summary


# --------------------------------------------------------------------
# 4)  Minimal CLI test (optional) -----------------------------------
if __name__ == "__main__":
    seg_paths = sorted(glob.glob("./BraTS2024-BraTS-GLI/training_data1_v2/BraTS-GLI*/BraTS-GLI*seg.nii.gz"))
    
    tumour_labels = {"ET": 3, "SNFH": 2, "NETC": 1, "RC": 4}
    atlas_overlap = {"ET": [], "SNFH": [], "NETC": [], "RC": []}
    #tumour_labels = {"ET": 3, "SNFH": 2, "NETC": 1}
    #atlas_overlap = {"ET": [], "SNFH": [], "NETC": []}
    for seg_path in tqdm(seg_paths):
        try:
            summ = analyze_label_localization(seg_path=seg_path, tumour_labels=tumour_labels, debug=False)
            for tumor_label, info in summ.items():
                if info['total_voxels'] > 0:
                    atlas_overlap[tumor_label].append(info['overlap_fraction']*100)
        except Exception as e:
            print(f"Error processing {seg_path}: {e}")
    print("\n\nSummary of atlas overlap percentages (%):")
    for tumor_label, overlaps in atlas_overlap.items():
        print(f"{tumor_label}: {np.mean(overlaps):.2f} ± {np.std(overlaps):.2f}, #samples: {len(overlaps)}")
