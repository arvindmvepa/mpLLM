import os
import pandas as pd
from tqdm import tqdm
import csv
import sys
import numpy as np
csv.field_size_limit(sys.maxsize)


def get_string_lst_statistics(text_lst):
    # Calculate the number of words in the rows
    lengths = np.array([count_words(text) for text in text_lst])

    # Calculate statistics
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)

    argmin_length_index = np.argmin(lengths)
    argmax_length_index = np.argmax(lengths)
    print("Shortest text:", text_lst[argmin_length_index])
    print("Longest text:", text_lst[argmax_length_index])

    print("Number of paragraphs: ", len(text_lst))
    print("Mean length:", mean_length)
    print("Standard deviation:", std_length)
    print("Minimum length:", min_length)
    print("Maximum length:", max_length)

    # Calculate quartiles and IQR
    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    iqr = q3 - q1

    # Define the upper whisker threshold
    upper_whisker = q3 + 1.5 * iqr

    # Identify outliers
    outliers = lengths[lengths > upper_whisker]

    if len(outliers) > 0:
        # Find the minimum outlier above the upper whisker
        min_outlier_above_max = np.min(outliers)
        print("Minimum outlier above the maximum:", min_outlier_above_max)
        print("Number of outliers above the upper whisker:", len(outliers))
    else:
        print("No outliers above the upper whisker.")


def get_words(text):
    return text.split()


def count_words(text):
    return len(get_words(text))


def load_csv_with_weird_bytes(file_path):
    # Read the file as bytes
    with open(file_path, 'rb') as f:
        csv_reader = csv.reader(f)

    rows = [row[0] for row in csv_reader]
    return rows


def split_csv_into_paras(input_csv_path, para_sep=".\n"):
    docs = load_csv_with_weird_bytes(input_csv_path)
    clean_paras = []
    for doc in docs:
        print(repr(doc))
        paras = doc.split(para_sep)
        print(len(paras))
        for para in paras:
            if para == "":
                continue
            clean_paras.append(clean_para(para.split("\n")))
    return clean_paras


def clean_para(text_lines):
    modified_lines = []
    for line in text_lines:
        if line == "":
            continue
        # remove hyphen that seperates words on different lines
        if line[-1] == "-":
            modified_lines.append(line[:-1])
        # add spaces between lines if there are no dashes
        else:
            modified_lines.append(line + " ")
    clean_para_string = "".join(modified_lines)
    # collapse multiple spaces into one
    clean_para_string = " ".join(get_words(clean_para_string))
    # remove null byte
    clean_para_string = clean_para_string.replace('\x00', '')
    return clean_para_string


def clean_text(text, para_sep=".\n", join_paras=True, remove_short_paras=False):
    """
    Clean and preprocess text by removing excessive whitespaces and optionally
    other non-standard formatting.
    """
    # Remove leading and trailing whitespaces
    text = text.strip()
    clean_paras = []
    # default split on period followed by newline
    paras = text.split(para_sep)
    for para in paras:
        para = para.strip()
        clean_para_ = clean_para(para.split("\n"))
        num_words = count_words(clean_para_)
        if (clean_para_ == "") or (remove_short_paras and num_words < 50):
            continue
        clean_paras.append(clean_para_)
    if join_paras:
        clean_paras = para_sep.join(clean_paras)
    return clean_paras


def generate_clean_paras(directory, remove_short_paras=True):
    data = []
    for filename in tqdm(sorted(os.listdir(directory))):
        if filename.endswith(".txt"):  # Adjust this condition based on your file types
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                paras = clean_text(text, join_paras=False, remove_short_paras=remove_short_paras)
                data.extend(paras)
    return data


def standard_save_to_csv(directory, output_csv_path, clean=True, save=True):
    data = []
    for filename in tqdm(sorted(os.listdir(directory))):
        if filename.endswith(".txt"):  # Adjust this condition based on your file types
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if clean:
                    text = clean_text(text)
                if text:
                    data.append([text])
    if save:
        df = pd.DataFrame(data, columns=["text"])
        df.to_csv(output_csv_path, index=False)
    return data


if __name__ == '__main__':
    chunk_size = 512
    directory = r"textbook_article_txt_files"
    output_csv_path = f"neph_v4.csv"
    save = False
    #data = standard_save_to_csv(directory, output_csv_path, save=save)
    #paras = split_csv_into_paras(output_csv_path)
    #print(len(paras))
    paras = generate_clean_paras(directory)
    print(len(paras))
    get_string_lst_statistics(paras)