import re
import csv
from utils.dataset_utils import clean_para, generate_clean_paras
import tiktoken


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_from_levenshtein(s1, s2):
    lev_distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0  # both strings are empty
    similarity = 1 - (lev_distance / max_len)
    return similarity


def jaccard_similarity(str1, str2):
    # Split the strings into tokens
    tokens1 = set(str1.split())
    tokens2 = set(str2.split())

    # Find the intersection and union of the two token sets
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    # Calculate Jaccard similarity
    if len(union) == 0:
        return 0  # Avoid division by zero
    jaccard_index = len(intersection) / len(union)
    return jaccard_index


def first_letter(s):
    m = re.search(r'[a-z(]', s, re.I)
    if m is not None:
        return m.start()
    return -1


def load_csv_answers(file_path):
    answers = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            # Assuming the "answer" column is the second column, index 1
            # Adjust the index as necessary for your specific CSV file
            answers.append(row[1])
    return answers


def open_file_and_clean_para(file_path):
    with open(file_path, 'r') as file:
        text_lines = file.read().split("\n")
    return clean_para(text_lines)


def clean_entity_list(file_contents_or_path, pred_in_str=False):
    if pred_in_str:
        entity_lst = file_contents_or_path.split("\n")
    else:
        with open(file_contents_or_path, 'r') as file:
            entity_lst = file.read().split("\n")
    modified_entity_lst = []
    for entity in entity_lst:
        if entity == "":
            continue
        # find the first relevant character and ignore other starting characters
        first_letter_index = first_letter(entity)
        mod_entity = entity[first_letter_index:].strip().lower()
        if len(mod_entity) < 50:
            modified_entity_lst.append(mod_entity)
    return modified_entity_lst


def annotate_entities_in_para(clean_para_string, clean_entity_list):
    start_index = 0
    for entity in clean_entity_list:
        modified_entity = entity.lower()
        lower_para_string = clean_para_string.lower()
        # find the entity index in the paragraph, starting from the previous entity
        find_index = lower_para_string[start_index:].find(modified_entity)
        if find_index > -1:
            entity_start_index = start_index + find_index
            entity_stop_index = entity_start_index + len(modified_entity)
            clean_para_string = clean_para_string[:entity_start_index] + "[" + \
                                clean_para_string[entity_start_index:entity_stop_index] + "]" + \
                                clean_para_string[entity_stop_index:]
            start_index = entity_stop_index + 2
        else:
            continue
    return clean_para_string

# TODO: add entity results into the relation extraction results
def generate_kg_prompts_from_dir(directory, prompt, tokenizer_name="cl100k_base", min_tokens=50,
                                 max_tokens=1024):
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    paras = generate_clean_paras(directory)
    mod_paras = []
    total_num_tokens = 0
    print("len(paras): ", len(paras))
    for para in paras:
        tokenized_ids = tokenizer.encode(para)
        if min_tokens <= len(tokenized_ids) <= max_tokens:
            mod_para = prompt + "\n" + para
            mod_paras.append(mod_para)
            mod_tokenized_ids = tokenizer.encode(mod_para)
            total_num_tokens += len(mod_tokenized_ids)
    return mod_paras, total_num_tokens


def get_full_prompts_and_calculate_token_lengths(directory, prompt, min_tokens=50, max_tokens=4096):
    results, total_num_tokens = generate_kg_prompts_from_dir(directory, prompt, min_tokens=min_tokens,
                                                             max_tokens=max_tokens)
    print("Number prompts: ", len(results))
    print("Total tokens: ", total_num_tokens)
    return results


def annotate_entities_in_files(p_file_path, e_file_path):
    para_string = clean_para(p_file_path)
    entity_list = clean_entity_list(e_file_path)
    ann_para_string = annotate_entities_in_para(para_string, entity_list)
    print("START")
    print(entity_list)
    print('-------------------------------------------------')
    print(para_string)
    print('-------------------------------------------------')
    print(ann_para_string)


def calculate_mention_metrics(gt_file, pred_contents_or_file, pred_in_str=False, thresh=0.5):
    gt_mentions = set(clean_entity_list(gt_file))
    print("gt_mentions: ", gt_mentions)
    pred_mentions = set(clean_entity_list(pred_contents_or_file, pred_in_str=pred_in_str))
    print(pred_mentions)
    total_mentions = gt_mentions.union(pred_mentions)
    tp = 0
    fp = 0
    fn = 0
    for mention in total_mentions:
        if (mention in gt_mentions) and (mention in pred_mentions):
            tp += 1
            gt_mentions.remove(mention)
            pred_mentions.remove(mention)
        elif (mention in gt_mentions) and (mention not in pred_mentions):
            match_found = False
            for mention_ in pred_mentions:
                if similarity_from_levenshtein(mention, mention_) >= thresh:
                    match_found = True
                    tp += 1
                    gt_mentions.remove(mention)
                    pred_mentions.remove(mention_)
                    break
            if not match_found:
                fn += 1
                gt_mentions.remove(mention)
        elif (mention not in gt_mentions) and (mention in pred_mentions):
            match_found = False
            for mention_ in gt_mentions:
                if similarity_from_levenshtein(mention, mention_) >= thresh:
                    match_found = True
                    tp += 1
                    gt_mentions.remove(mention_)
                    pred_mentions.remove(mention)
                    break
            if not match_found:
                fp += 1
                pred_mentions.remove(mention)
    return tp, fp, fn, len(total_mentions)


def calculate_mention_metrics_from_files(gt_files, pred_files=None, csv_file=None):
    assert (pred_files or csv_file) and not (pred_files and csv_file), "One of pred_contents or csv_file should be provided"
    if csv_file:
        pred_contents = load_csv_answers(csv_file)
        pred_contents_or_files = pred_contents
    else:
        pred_contents_or_files = pred_files
    mean_f1_score = 0
    mean_prec_score = 0
    mean_recall_score = 0
    for gt_file, pred_contents_or_file in zip(gt_files, pred_contents_or_files):
        tp,fp,fn,total = calculate_mention_metrics(gt_file, pred_contents_or_file, pred_in_str=csv_file is not None, thresh=0.7)
        f1_score = 2 * tp / (2 * tp + fp + fn + 1e-6)
        recall_score = tp/ (tp + fn + 1e-6)
        prec_score = tp / (tp + fp + 1e-6)
        print(f"f1_score: {f1_score}, prec_score : {prec_score}, recall_score: {recall_score}, tp: {tp}, fp: {fp}, fn: {fn}, total: {total}")
        mean_f1_score += f1_score
        mean_prec_score += prec_score
        mean_recall_score += recall_score
    mean_f1_score /= len(gt_files)
    mean_prec_score /= len(gt_files)
    mean_recall_score /= len(gt_files)
    print(f"mean_f1_score: {mean_f1_score}, mean_prec_score : {mean_prec_score}, mean_recall_score: {mean_recall_score}")


if __name__ == "__main__":
    # PERFORMS THE ENTITY/RELATION EXTRACTION TASK ON THE DATASET
    directory = r"textbook_article_txt_files"
    # prompt for relation extraction
    prompt = "Extract all the relations from the provided paragraph. The list of possible relation types are provided below:\n\n" \
    "isa\nassociated_with\n physically_related_to\n  part_of\n  consists_of\n  contains\n  connected_to\n  interconnects\n  branch_of\n  tributary_of\n  ingredient_of\n" \
    " spatially_related_to\n  location_of\n  adjacent_to\n  surrounds\n  traverses\n functionally_related_to\n" \
    "  affects\n   manages\n   treats\n   disrupts\n   complicates\n   interacts_with\n   prevents\n  brings_about\n   produces\n   causes\n" \
    "  performs\n   carries_out\n   exhibits\n   practices\n  occurs_in\n   process_of\n  uses\n  manifestation_of\n  indicates\n  result_of\n" \
    " temporally_related_to\n  co-occurs_with\n  precedes\n conceptually_related_to\n  evaluation_of\n  degree_of\n  analyzes\n   assesses_effect_of\n" \
    "  measurement_of\n  measures\n  diagnoses\n  property_of\n  derivative_of\n  developmental_form_of\n  method_of\n  conceptual_part_of\n  issue_in\n\n" \
    "You may add relations not listed if necessary. Each entity is marked in the paragraph with brackets []: however, "\
    "these brackets may not enclose the full entity or may be missing entities so please adjust as necessary. Detect "\
    "the most applicable relation between entities and provide a list of all the detected tuples in the form: "\
    "(entity, relation, entity). If there are multiple entities before or after the relation, please separate them by "\
    "\"AND\" or \"OR\" depending on context. An example paragraph and generated relation tuples are provided below:\n\n"\
    "[Glomerular hypertrophy] may be marker of [FSGS]. [Glomerular enlargement] precedes [overt glomerulosclerosis] "\
    "in FSGS (19). [Patients] with [abnormal glomerular growth] on initial [biopsies] that did not show "\
    "[overt sclerotic lesions] subsequently developed [overt glomerulosclerosis], documented in later [biopsies]." \
    "A cutoff of [glomerular area] larger than 50%  more than normal for " \
    "[age] indicated increased risk for [progression]. Of note, [glomeruli] grow until approximately age 18 years, "\
    "so [age-matched controls] must be used in the [pediatric  population].  Since  [tissue  processing  methods]  "\
    "may inﬂuence the size of [structures in tissue], it is imperative that each [laboratory] determines "\
    "[normal ranges] for this [parameter].\n\n" \
    "(Glomerular hypertrophy, indicates, FSGS)\n" \
    "(Glomerular enlargement, precedes, overt glomerulosclerosis)\n"\
    "(Glomerular enlargement, occurs_in, FSGS)\n"\
    "(overt glomerulosclerosis, occurs_in, FSGS)\n"\
    "(Patients AND biopsies, manifestation_of, abnormal glomerular growth)\n"\
    "(overt sclerotic lesions, prevents, overt glomerulosclerosis)\n"\
    "(abnormal glomerular growth, produces, biopsies AND overt glomerulosclerosis)\n"\
    "(glomerular area AND larger than 50 % more than normal for age, indicates, progression)\n"\
    "(glomeruli, measured by, age-matched controls AND pediatric population)\n"\
    "(tissue processing methods, interacts_with, size of structures in tissue)\n"\
    "(laboratory, measures, normal ranges AND size of structures in tissue)\n\n"\
    "Please perform the extraction on the following paragraph:"
    """
    # prompt for entity extraction
    prompt = "Extract all the entities from the provided paragraph. Please provide them in order in a list format. An example is provided below.\n\n"\
             "Glomerular hypertrophy may be marker of FSGS. Glomerular enlargement precedes overt glomerulosclerosis in FSGS (19). "\
             "Patients with abnormal glomerular growth on initial biopsies that did not show overt sclerotic " \
             "lesions subsequently developed overt glomerulosclerosis, documented in" \
             "later biopsies.\n\n" \
             "Glomerular hypertrophy\n" \
             "FSGS\n" \
             "Glomerular enlargement\n" \
             "overt glomerulosclerosis\n" \
             "patients\n" \
             "abnormal glomerular growth\n" \
             "biopsies\n" \
             "overt sclerotic lesions\n" \
             "overt glomerulosclerosis\n" \
             "biopsies\n\n" \
             "Please perform the extraction on the following paragraph:"
    """
    #get_full_prompts_and_calculate_token_lengths(directory, prompt)

    """
    # CALCULATES THE MENTION DETECTION SCORES
    p_file_path = r'/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/paragraphs/p2_HFHS_CKD_V6.txt'
    e_file_path = r'/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p2_results/entities/p2_HFHS_CKD_V6_chatgpt3.5_gt_entities_prompt2.txt'
    annotate_entities_in_files(p_file_path, e_file_path)
    """

    gt_files = [r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p2/entities/p2_HFHS_CKD_V6_gt_entities_neph_kg_clean.txt",
                r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p3/entities/p3_Brenner__Rector's_The_Kidney_-_47th_60_entities_gt_neph_kg_clean.txt",
                r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p4/entities/p4_adk5-15_entities_gt_neph_kg_clean.txt",
                r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p5/entities/p5_Bookshelf_NBK51773_entities_gt_neph_kg_clean.txt",
                r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p6/entities/p6_surgical-aspects-of-kidney-transplantation_entities_gt_neph_kg_clean.txt"]
    """
    pred_files = [r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p2/entities/p2_HFHS_CKD_V6_prompt_eng_chatgpt3.5.txt",
                  r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p3/entities/p3_Brenner__Rector's_The_Kidney_-_47th_60_prompt_eng_chatgpt3.5.txt",
                  r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p4/entities/p4_adk5-15_prompt_eng_chatgpt3.5.txt",
                  r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p5/entities/p5_Bookshelf_NBK51773_prompt_eng_chatgpt3.5.txt",
                  r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p6/entities/p6_surgical-aspects-of-kidney-transplantation_prompt_eng_chatgpt3.5.txt"]
    pred_files = [r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p2/entities/p2_HFHS_CKD_V6_prompt_eng_chatgpt4.txt",
                  r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p3/entities/p3_Brenner__Rector's_The_Kidney_-_47th_60_prompt_eng_chatgpt4.txt",
                  r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p4/entities/p4_adk5-15_prompt_eng_chatgpt4.txt",
                  r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p5/entities/p5_Bookshelf_NBK51773_prompt_eng_chatgpt4.txt",
                  r"/Users/arvin/Documents/ucla research/nephrology nlp/kg extraction tests/p6/entities/p6_surgical-aspects-of-kidney-transplantation_prompt_eng_chatgpt4.txt"]
    """
    #csv_file = r"HuggingFaceH4_zephyr-7b-beta_eval.csv"
    #csv_file = "llama2_7b_eval.csv"
    #csv_file = "llama3_8b_eval.csv"
    #csv_file = "pmc_llama_13b_eval.csv"
    csv_file = "openai_results_gpt3.csv"
    pred_files = None
    calculate_mention_metrics_from_files(gt_files, pred_files, csv_file=csv_file)