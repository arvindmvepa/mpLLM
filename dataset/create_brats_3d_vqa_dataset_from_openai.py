import itertools, random
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from vqa_3d_utils import convert_entry


base_types = [1, 2, 3, 4]
base_type_mapping = {1: "area", 2: "region", 3: "shape", 4: "satellite"}
unknown_type = 5
all_combos = [
    tuple(sorted(c))
    for r in range(1, len(base_types) + 1)
    for c in itertools.combinations(base_types, r)
]

combo_question_type_map = {(1,): "Q: How large is the volume covered by {label}? A: The overall volume of {label} is {area}.",
                           (2,): "Q: Which region(s) of the brain is {label} located in? A: The {label} is located in {regions}.",
                           (3,): "Q: What is the shape of {label}? A: The shape of {label} is {shape}.",
                           (4,): "Q: How spread out is {label}? A: The spread of {label} is {satellite}.",
                           (1, 2): "Q: How large is the volume of {label} and where is it located? A: The overall volume of {label} is {area}, and it is located in {regions}.",
                           (1, 3): "Q: How large is the volume of {label} and what is its shape? A: The overall volume of {label} is {area}, and its shape is described as {shape}.",
                           (1, 4): "Q: How large is the volume of {label} and how spread out is it? A: The overall volume of {label} is {area}, and it is characterized as {satellite}.",
                           (2, 3): "Q: In which region is {label} and what is its shape? A: The {label} is located in {regions}, and its shape is described as {shape}.",
                           (2, 4): "Q: In which region is {label} and how spread out is it? A: The {label} is located in {regions}, and it is characterized as {satellite}.",
                           (3, 4): "Q: What is the shape of {label} and how spread out is it? A: The shape of {label} is described as {shape}, and it is characterized as {satellite}.",
                           (1, 2, 3): "Q: What is the volume, region, and shape of {label}? A: The overall volume of {label} is {area}, it is located in {regions}, and its shape is described as {shape}.",
                           (1, 2, 4): "Q: What is the volume, region, and spread of {label}? A: The overall volume of {label} is {area}, it is located in {regions}, and it is characterized as {satellite}.",
                           (1, 3, 4): "Q: What is the volume, shape, and spread of {label}? A: The overall volume of {label} is {area}, its shape is described as {shape}, and it is characterized as {satellite}.",
                           (2, 3, 4): "Q: What is the region, shape, and spread of {label}? A: The {label} is located in {regions}, its shape is described as {shape}, and it is characterized as {satellite}.",
                           (1, 2, 3, 4): "Q: What is the volume, region, shape, and spread of {label}? A: The overall volume of {label} is {area}, it is located in {regions}, its shape is described as {shape}, and it is characterized as {satellite}."}
question_type_combo_map = {v: k for k, v in combo_question_type_map.items()}


def validate_vqa_lists(vqa_list, save_dir=None):
    """
    Compute and (optionally) persist statistics that sanity‑check your VQA data.

    Parameters
    ----------
    vqa_list : list[dict]
        Each element must contain at least these keys
            question, answer, label_name, type, combo
        where
            question : str   – rendered "Q: …"
            answer   : str   – rendered "A: …"
            label_name : str – e.g. "Enhancing"
            type       : str – one of {"area","region","shape","satellite"}
            combo      : tuple[int] – e.g. (1, 3, 4)
    save_dir : str | Path | None, default None
        If provided, CSV versions of the tables are written there.

    Returns
    -------
    dict[str, pd.DataFrame]  – the six summary tables for further inspection.
    """
    QUESTION_COL = "question"
    ANSWER_COL = "answer"
    LABEL_COL = "label_name"
    TYPE_COL = "type"
    COMBO_COL = "combo"

    required_cols = {QUESTION_COL, ANSWER_COL, LABEL_COL, TYPE_COL, COMBO_COL}

    df = pd.DataFrame(vqa_list)
    print(f"Loaded {len(df):,} rows  ({len(vqa_list):,})")

    # Basic schema check
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Each VQA dict must include {sorted(required_cols)}. Missing: {missing}")

    df[COMBO_COL] = df[COMBO_COL].apply(
        lambda c: tuple(c) if isinstance(c, list) else c
    )
    combo_overall = (df[COMBO_COL]
          .value_counts()
          .rename_axis("combo")
          .reset_index(name="n_questions")
          .sort_values("combo", key=lambda s: s.apply(str))
    )

    combo_per_label = (
        df.groupby([LABEL_COL, COMBO_COL])
          .size()
          .rename("n_questions")
          .reset_index()
    )

    combo_per_label_type = (
        df.groupby([LABEL_COL, TYPE_COL, COMBO_COL])
          .size()
          .rename("n_questions")
          .reset_index()
    )

    # ─────────────────────────────────────────────────────────────
    # 4.  Unique‑question / unique‑answer counts
    # ─────────────────────────────────────────────────────────────
    unique_q_overall = df[QUESTION_COL].nunique()
    unique_a_overall = df[ANSWER_COL].nunique()

    unique_q_per_label = (
        df.groupby(LABEL_COL)[QUESTION_COL]
          .nunique()
          .rename("n_unique_questions")
          .reset_index()
    )

    unique_a_per_label = (
        df.groupby(LABEL_COL)[ANSWER_COL]
          .nunique()
          .rename("n_unique_answers")
          .reset_index()
    )

    unique_q_per_label_type = (
        df.groupby([LABEL_COL, TYPE_COL])[QUESTION_COL]
          .nunique()
          .rename("n_unique_questions")
          .reset_index()
    )

    unique_a_per_label_type = (
        df.groupby([LABEL_COL, TYPE_COL])[ANSWER_COL]
          .nunique()
          .rename("n_unique_answers")
          .reset_index()
    )

    print("\n▶ Combo distribution (overall)")
    print(combo_overall.to_string(index=False))

    print("\n▶ Unique Q/A counts (overall)")
    print(f"   • questions : {unique_q_overall:,}")
    print(f"   • answers   : {unique_a_overall:,}")

    # ─────────────────────────────────────────────────────────────
    # 6.  Optional CSV export
    # ─────────────────────────────────────────────────────────────
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        combo_overall.to_csv(save_dir / "combo_overall.csv", index=False)
        combo_per_label.to_csv(save_dir / "combo_per_label.csv", index=False)
        combo_per_label_type.to_csv(save_dir / "combo_per_label_type.csv", index=False)
        unique_q_per_label.to_csv(save_dir / "unique_q_per_label.csv", index=False)
        unique_a_per_label.to_csv(save_dir / "unique_a_per_label.csv", index=False)
        unique_q_per_label_type.to_csv(save_dir / "unique_q_per_label_type.csv", index=False)
        unique_a_per_label_type.to_csv(save_dir / "unique_a_per_label_type.csv", index=False)

        print(f"\nCSV tables written to → {save_dir.resolve()}")

    # ─────────────────────────────────────────────────────────────
    # 7.  Return tables for programmatic inspection
    # ─────────────────────────────────────────────────────────────
    return {
        "combo_overall": combo_overall,
        "combo_per_label": combo_per_label,
        "combo_per_label_type": combo_per_label_type,
        "unique_q_per_label": unique_q_per_label,
        "unique_a_per_label": unique_a_per_label,
        "unique_q_per_label_type": unique_q_per_label_type,
        "unique_a_per_label_type": unique_a_per_label_type,
    }


def map_df_cols_to_combo(df):
    # iterate over the rows of the dataframe
    for i, row in df.iterrows():
        # get the combo for the current row
        original_qa_prompt = row["original_qa"][row["original_qa"].index("Q: "):]
        combo = question_type_combo_map[original_qa_prompt]
        df.at[i, "combo"] = str(combo)
    return df


def map_df_cols_to_combo_and_unknown(df):
    # iterate over the rows of the dataframe
    for i, row in df.iterrows():
        # get the combo for the current row
        original_qa_prompt = row["original_qa"][row["original_qa"].index("Q: "):]
        combo = question_type_combo_map[original_qa_prompt] + (unknown_type,)
        df.at[i, "combo"] = str(combo)
    return df


def validate_question_answer_combo(question, answer_template, combo):
    flag = True
    flag = flag and "{label}" in question
    if ("{area}" in answer_template) or (1 in combo):
        flag = flag and (("{area}" in answer_template) and 1 in combo)
    if ("{regions}" in answer_template or "{region}" in answer_template) or (2 in combo):
        flag = flag and (("{regions}" in answer_template or "{region}" in answer_template) and (2 in combo))
    if ("{shape}" in answer_template) or (3 in combo):
        flag = flag and (("{shape}" in answer_template) and (3 in combo))
    if ("{satellite}" in answer_template) or (4 in combo):
        flag = flag and (("{satellite}" in answer_template) and (4 in combo))
    return flag


def map_df_cols_to_unknown(df):
    # map all rows to unknown type
    for i, row in df.iterrows():
        combo = (unknown_type,)
        df.at[i, "combo"] = str(combo)
    return df


def pick_question_from_df(df):
    row = df.iloc[0]
    temp_question = row["transformed_q"]
    temp_answer = row["transformed_a"]
    temp_combo_str = row["combo"]
    temp_combo = tuple([int(num) for num in temp_combo_str.strip("()").split(",") if len(num) > 0])
    row_idx = row.name
    df.drop(row_idx, inplace=True)
    while not validate_question_answer_combo(temp_question, temp_answer, temp_combo):
        print(f"Invalid question/answer combo: {temp_question}, {temp_answer}, {temp_combo}")
        # TODO: check for length of filt_df to make sure there are valid rows left
        row = df.iloc[0]
        temp_question = row["transformed_q"]
        temp_answer = row["transformed_a"]
        temp_combo_str = row["combo"]
        temp_combo = tuple([int(num) for num in temp_combo_str.strip("()").split(",") if len(num) > 0])
        row_idx = row.name
        df.drop(row_idx, inplace=True)
    question = temp_question
    answer = temp_answer
    combo = temp_combo
    return question, answer, combo


def pick_num_question_types_combos_and_rows(df, rng):
    shuffled_base_types = base_types[:]
    rng.shuffle(shuffled_base_types)
    shuffled_combos = all_combos[:]
    rng.shuffle(shuffled_combos)

    used_combos = list()
    qas = {}

    for t in shuffled_base_types:
        # make sure we pick combos that contain the current type and haven't been used previously
        filtered_shuffled_combos = [combo for combo in shuffled_combos if (t in combo) and (combo not in used_combos)]
        for temp_combo in filtered_shuffled_combos:
            question = None
            answer = None
            combo = None
            temp_str_combo = str(tuple(temp_combo))
            while len(df[df["combo"] == temp_str_combo]) > 0:
                filt_df = df[df["combo"] == temp_str_combo]
                row = filt_df.iloc[0]
                temp_question = row["transformed_q"]
                temp_answer = row["transformed_a"]
                row_idx = row.name
                df.drop(row_idx, inplace=True)
                if validate_question_answer_combo(temp_question, temp_answer, temp_combo):
                    question = temp_question
                    answer = temp_answer
                    combo = temp_combo
                    used_combos.append(combo)
                    question_type = base_type_mapping[t]
                    qas[question_type] = (question, answer, combo)
                    break
                else:
                    print(f"Invalid question/answer combo: {temp_question}, {temp_answer}, {temp_combo}")
                    continue
            # break if a valid question/answer/combo was found; otherwise look at other combos
            if (question is not None) and (answer is not None) and (combo is not None):
                break
        else:
            raise ValueError(
                f"No available question containing type {t} for this pair."
            )
    assert len(qas) == len(shuffled_base_types), f"qas: {qas}, shuffled_base_types: {shuffled_base_types}"
    return qas


def organize_vqa_data_by_seg_id_and_label_and_type(vqa_data, question_key="volume_file_id", type_key="type",
                                                   label_key="label_name", add_partially_unknown=True,
                                                   add_unknown=True):
    vqa_data_dict = dict()
    for vqa_datum in vqa_data:
        seg_id = vqa_datum[question_key]
        label = vqa_datum[label_key]
        question_type = vqa_datum[type_key]
        if seg_id not in vqa_data_dict:
            vqa_data_dict[seg_id] = {}
        if label not in vqa_data_dict[seg_id]:
            vqa_data_dict[seg_id][label] = {}
        vqa_data_dict[seg_id][label][question_type] = vqa_datum
        if add_partially_unknown:
            # just add the last datum as the partially unknown question as a placeholder
            if "partially_unknown" not in vqa_data_dict[seg_id][label]:
                new_vqa_datum = dict(vqa_datum)
                new_vqa_datum[type_key] = "unknown"
                new_vqa_datum["content_type"] = "partially_unknown"
                vqa_data_dict[seg_id][label]["partially_unknown"] = new_vqa_datum
        if add_unknown:
            # just add the last datum as the unknown question as a placeholder
            if "unknown" not in vqa_data_dict[seg_id][label]:
                new_vqa_datum = dict(vqa_datum)
                new_vqa_datum[type_key] = "unknown"
                new_vqa_datum["content_type"] = "unknown"
                vqa_data_dict[seg_id][label]["unknown"] = new_vqa_datum
    return vqa_data_dict


def unorganize_vqa_data_by_seg_id_and_label_and_type(vqa_data):
    vqa_data_list = []
    for seg_id, seg_id_vqa_datum in vqa_data.items():
        for label, label_vqa_datum in seg_id_vqa_datum.items():
            for question_type, vqa_datum in label_vqa_datum.items():
                vqa_data_list.append(vqa_datum)
    return vqa_data_list


def generate_updated_vqa_data(vqa_data_dict, seed, openai_df, openai_partially_unknown_df=None, openai_unknown_df=None,
                              question_types=("area", "region", "shape", "satellite", "partially_unknown", "unknown")):
    rng = random.Random(seed)
    for seg_id, labels_question_types_vqa_datum in tqdm(vqa_data_dict.items()):
        for label, question_types_vqa_datum in labels_question_types_vqa_datum.items():

            # ---- RESET PER-LABEL PLACEHOLDER VALUES ----
            area = regions = shape = satellite = None

            # -------------------------------------------
            qas = pick_num_question_types_combos_and_rows(df=openai_df, rng=rng)

            # collect the per-label ground-truth answers
            for question_type in question_types:
                vqa_datum = question_types_vqa_datum[question_type]
                ans = vqa_datum["answer_vqa"]
                if question_type == "area": area = ans[0]
                elif question_type == "region": regions = ans[0]
                elif question_type == "shape": shape = ans[0]
                elif question_type == "satellite": satellite = ans[0]

            # sample extra question types
            if openai_partially_unknown_df is not None:
                q, a, combo = pick_question_from_df(openai_partially_unknown_df)
                qas["partially_unknown"] = (q, a, combo)
            if openai_unknown_df is not None:
                q, a, combo = pick_question_from_df(openai_unknown_df)
                qas["unknown"] = (q, a, combo)

            # now update each datum
            for question_type in question_types:
                vqa_datum = question_types_vqa_datum[question_type]
                question, answer_template, combo = qas[question_type]

                question = question.replace("{label}", label)
                filled_answer = answer_template.replace("{label}", label)

                answer_vqa = []
                if ("{area}" in answer_template) or (1 in combo):
                    assert ("{area}" in answer_template) and 1 in combo, f"question: {question}, answer: {answer_template}, combo: {combo}"
                    filled_answer = filled_answer.replace("{area}", area)
                    answer_vqa.append([area])
                if ("{regions}" in answer_template or "{region}" in answer_template) or (2 in combo):
                    assert ("{regions}" in answer_template or "{region}" in answer_template) and (2 in combo), f"question: {question}, answer: {answer_template}, combo: {combo}"
                    if "{regions}" in filled_answer:
                        filled_answer = filled_answer.replace("{regions}", regions)
                    if "{region}" in filled_answer:
                        filled_answer = filled_answer.replace("{region}", regions)
                    answer_vqa.append([regions])
                if ("{shape}" in answer_template) or (3 in combo):
                    assert ("{shape}" in answer_template) and (3 in combo), f"question: {question}, answer: {answer_template}, combo: {combo}"
                    filled_answer = filled_answer.replace("{shape}", shape)
                    answer_vqa.append([shape])
                if ("{satellite}" in answer_template) or (4 in combo):
                    assert ("{satellite}" in answer_template) and (4 in combo), f"question: {question}, answer: {answer_template}, combo: {combo}"
                    filled_answer = filled_answer.replace("{satellite}", satellite)
                    answer_vqa.append([satellite])
                if (question_type in {"partially_unknown", "unknown"}) or (5 in combo):
                    assert (question_type in {"partially_unknown", "unknown"}) and (5 in combo), f"question: {question}, answer: {answer_template}, combo: {combo}"
                    answer_vqa.append(["unknown"])

                vqa_datum.update(
                    question=question,
                    answer=filled_answer,
                    answer_vqa=answer_vqa,   # flat list of strings
                    answer_gen=filled_answer,
                    combo=combo,
                    type=question_type,
                    content_type=question_type,
                )

    return vqa_data_dict


if __name__ == "__main__":
    ref_train_vqa_file = "brats_{}_3d_vqa_subj{}_train_{}.json"
    ref_val_vqa_file = "brats_{}_3d_vqa_subj{}_val_{}.json"
    ref_test_vqa_file = "brats_{}_3d_vqa_subj{}_test_{}.json"

    train_vqa_file = "brats_{}_3d_vqa_subj{}_train_{}_multitask_fixed.json"
    val_vqa_file = "brats_{}_3d_vqa_subj{}_val_{}_multitask_fixed.json"
    test_vqa_file = "brats_{}_3d_vqa_subj{}_test_{}_multitask_fixed.json"

    openai_df_file = "mri_dataset.csv"
    openai_partially_unknown_df_file = "mri_dataset_partially_unknown.csv"
    openai_unknown_df_file = "mri_dataset_unknown.csv"
    # rest of the parameters
    subjective_only = True
    dataset_seed = 0
    new_dataset_seed = 0

    # GLI dataset settings
    dataset_type = "gli"
    version = f"updated_v2_seed{dataset_seed}"
    labels_order = (1, 2, 3, 4)
    pediatric = False
    goat = False

    # MET dataset settings
    #dataset_type = "met"
    #version = f"updated_v3_seed{dataset_seed}"
    #labels_order = (1, 2, 3)
    #pediatric = False
    #goat = False

    # GoAT dataset settings
    #dataset_type = "goat"
    #version = f"updated_v3_seed{dataset_seed}"
    #labels_order = (1, 2, 3)
    #pediatric = False
    #goat = True

    ref_train_vqa_file = ref_train_vqa_file.format(dataset_type, subjective_only, version)
    ref_val_vqa_file = ref_val_vqa_file.format(dataset_type, subjective_only, version)
    ref_test_vqa_file = ref_test_vqa_file.format(dataset_type, subjective_only, version)

    train_vqa_file = train_vqa_file.format(dataset_type, subjective_only, version)
    val_vqa_file = val_vqa_file.format(dataset_type, subjective_only, version)
    test_vqa_file = test_vqa_file.format(dataset_type, subjective_only, version)

    question_key = "volume_file_id"
    type_key = "type"
    with open(ref_train_vqa_file, 'r') as f:
        ref_train_vqa_data = json.load(f)
        ref_train_vqa_data_dict = organize_vqa_data_by_seg_id_and_label_and_type(ref_train_vqa_data,
                                                                                 add_partially_unknown=openai_partially_unknown_df_file is not None,
                                                                                 add_unknown=openai_unknown_df_file is not None)
    with open(ref_val_vqa_file, 'r') as f:
        ref_val_vqa_data = json.load(f)
        ref_val_vqa_data_dict = organize_vqa_data_by_seg_id_and_label_and_type(ref_val_vqa_data,
                                                                               add_partially_unknown=openai_partially_unknown_df_file is not None,
                                                                               add_unknown=openai_unknown_df_file is not None)
    with open(ref_test_vqa_file, 'r') as f:
        ref_test_vqa_data = json.load(f)
        ref_test_vqa_data_dict = organize_vqa_data_by_seg_id_and_label_and_type(ref_test_vqa_data,
                                                                                add_partially_unknown=openai_partially_unknown_df_file is not None,
                                                                                add_unknown=openai_unknown_df_file is not None)

    # read the openai df files and map the combos
    openai_df = pd.read_csv(openai_df_file, header=0)
    openai_df = map_df_cols_to_combo(openai_df)
    openai_df = openai_df.sample(frac=1, random_state=new_dataset_seed)

    openai_partially_unknown_df = pd.read_csv(openai_partially_unknown_df_file, header=0)
    openai_partially_unknown_df = map_df_cols_to_combo_and_unknown(openai_partially_unknown_df)
    openai_partially_unknown_df = openai_partially_unknown_df.sample(frac=1, random_state=new_dataset_seed)

    openai_unknown_df = pd.read_csv(openai_unknown_df_file, header=0)
    openai_unknown_df = map_df_cols_to_unknown(openai_unknown_df)
    openai_unknown_df = openai_unknown_df.sample(frac=1, random_state=new_dataset_seed)

    # generate updated vqa dataset
    train_vqa_data_dict = generate_updated_vqa_data(ref_train_vqa_data_dict,
                                                    openai_df=openai_df,
                                                    openai_partially_unknown_df=openai_partially_unknown_df,
                                                    openai_unknown_df=openai_unknown_df,
                                                    seed=new_dataset_seed)
    train_vqa = unorganize_vqa_data_by_seg_id_and_label_and_type(train_vqa_data_dict)
    val_vqa_data_dict = generate_updated_vqa_data(ref_val_vqa_data_dict,
                                                  openai_df=openai_df,
                                                  openai_partially_unknown_df=openai_partially_unknown_df,
                                                  openai_unknown_df=openai_unknown_df,
                                                  seed=new_dataset_seed)
    val_vqa = unorganize_vqa_data_by_seg_id_and_label_and_type(val_vqa_data_dict)
    test_vqa_data_dict = generate_updated_vqa_data(ref_test_vqa_data_dict,
                                                   openai_df=openai_df,
                                                   openai_partially_unknown_df=openai_partially_unknown_df,
                                                   openai_unknown_df=openai_unknown_df,
                                                   seed=new_dataset_seed)
    test_vqa = unorganize_vqa_data_by_seg_id_and_label_and_type(test_vqa_data_dict)

    # create numeric entries for vqa
    for vqa_datum in train_vqa:
        vqa_datum["answer_vqa_numeric"] = convert_entry(vqa_datum,
                                                        add_unknown=(openai_partially_unknown_df_file is not None) or (openai_unknown_df_file is not None))
    for vqa_datum in val_vqa:
        vqa_datum["answer_vqa_numeric"] = convert_entry(vqa_datum,
                                                        add_unknown=(openai_partially_unknown_df_file is not None) or (openai_unknown_df_file is not None))
    for vqa_datum in test_vqa:
        vqa_datum["answer_vqa_numeric"] = convert_entry(vqa_datum,
                                                        add_unknown=(openai_partially_unknown_df_file is not None) or (openai_unknown_df_file is not None))
    # save the updated vqa dataset
    with open(train_vqa_file, 'w') as f:
        json.dump(train_vqa, f, indent=2)
    with open(val_vqa_file, 'w') as f:
        json.dump(val_vqa, f, indent=2)
    with open(test_vqa_file, 'w') as f:
        json.dump(test_vqa, f, indent=2)
