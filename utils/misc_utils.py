import yaml
import pandas as pd
import os
import json
import csv


def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def assert_required_params_list(required_params_lst, included_params_lst, header=""):
    def format(string):
        if header:
            return f"{header}: {string}"
        else:
            return string
    missing_sections = list_difference(required_params_lst, included_params_lst)
    assert len(missing_sections) == 0, format(f"missing required sections: {missing_sections}")
    extra_sections = list_difference(included_params_lst, required_params_lst)
    assert len(extra_sections) == 0, format(f"extra sections: {extra_sections}")


def list_difference(list1, list2):
   return [item for item in list1 if item not in list2]


def load_mrconso_as_df(umls_meta_path="2024AA-full/2024AA/META", ref_file="MRCONSO.RRF",
                       cui_col=0, lang_col=1, lang="ENG", term_col=14, combine_by_cui=False, combine_split_str=',,,',
                       save_path="mrconso.csv"):
    ref_path = os.path.join(umls_meta_path, ref_file)
    df = pd.read_csv(ref_path, delimiter='|', header=None, dtype=str)
    # filter by language
    df = df.loc[df[lang_col] == lang]
    # remove extra columns
    df = df[[cui_col, term_col]]
    if combine_by_cui:
        df[term_col] = df[term_col].apply(lambda x: [x])
        df = df.groupby(cui_col).sum().reset_index()
        # join lists by ',,,' to make it easier to split later
        df[term_col] = df[term_col].apply(lambda x: combine_split_str.join(map(str, x)))
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def groundtruth_csv_to_json(csv_file_path, json_file_path):
    data = []
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for idx, row in enumerate(csv_reader):
            row['qid'] = idx
            row['answer'] = row['clean answer']
            data.append(row)
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    csv_file_path = r"/Users/admin/Documents/research/nephrology nlp/groundtruth.csv"
    json_file_path = r"/Users/admin/Documents/research/nephrology nlp/groundtruth.json"
    groundtruth_csv_to_json(csv_file_path, json_file_path)
