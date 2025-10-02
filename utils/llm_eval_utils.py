import pandas as pd
import re
import json


def grade_llm_output(pred_file, gt_file, na_string="NA"):
    pred_df = pd.read_csv(pred_file)
    gt_df = pd.read_csv(gt_file)

    acc_with_no_na = 0.0
    acc_with_zero_na = 0.0
    num_w_na = 0
    num_no_na = 0

    # search thru gt df to find matching question in pred df
    for gt_index, gt_row in gt_df.iterrows():
        gt_question = gt_row['question']
        gt = gt_row['clean answer']

        if pd.isnull(gt_question) or pd.isnull(gt) or gt == na_string:
            continue

        # find matching pred df row or skip question
        pred = None
        for pred_index, pred_row in pred_df.iterrows():
            pred_row = pred_df.iloc[pred_index]
            pred_question = pred_row['question']
            if gt_question in pred_question:
                pred = pred_row['updated_answer']
                break

        # calculate acc w/ and w/o NA
        if pred == na_string or pd.isnull(pred):
            acc_with_zero_na += 0
            num_w_na += 1
            continue

        pred = pred.upper()
        gt = gt.upper()

        acc = pred == gt
        acc_with_no_na += acc
        acc_with_zero_na += acc
        num_no_na += 1
        num_w_na += 1

    acc_with_no_na /= num_no_na
    acc_with_zero_na /= num_w_na

    results = {"num_no_na": num_no_na, "acc_with_no_na": acc_with_no_na, "num_w_na": num_w_na, "acc_with_zero_na": acc_with_zero_na}

    return results


def collect_mc_answers_from_llm_output(answer_file, save_file, old_pred_key="old_answer", pred_key="model_answer"):
    with open(answer_file, "r") as f:
        questions = json.load(f)

    # Initialize total questions
    total_questions = len(questions)

    collected_answers = []
    for index, question in enumerate(questions):
        if old_pred_key in question:
            answer = question[old_pred_key]
        else:
            answer = question[pred_key]
        question[old_pred_key] = answer

        answer = answer.upper().strip()

        if len(answer) < 1:
            collected_answers.append("NA")
            question[pred_key] = "NA"
            continue

        # Check for matches
        current_answer = None
        choice_candidates = [['A', 'a', '1'], ['B', 'b', '2'], ['C', 'c', '3'], ['D', 'd', '4'], ['E', 'e', '5'],
                             ['F', 'f', '6'], ['G', 'g', '7'], ['H', 'h', '8'], ['I', 'i', '9']]
        for pattern in llm_regex_patterns:
            match = re.search(pattern, answer)
            if match:
                v = match.group(1)
                v = v.upper().strip()
                current_answer = [choice_candidate[0] for choice_candidate in choice_candidates if v in choice_candidate]
                if len(current_answer) > 0:
                    current_answer = current_answer[0]
                    question[pred_key] = current_answer
                    collected_answers.append(current_answer)
                    break
                else:
                    current_answer = None
        if current_answer is None:
            collected_answers.append("NA")
            question[pred_key] = "NA"

    with open(save_file, mode='w', encoding='utf-8') as json_file:
        json.dump(questions, json_file, indent=4)

    return total_questions, collected_answers


llm_regex_patterns = [
    # r"The most appropriate management for this patient is observation with expectant medical management \(option ([A-I])\)\.:?",
    # r"option ([A-I]).*is the correct answer",

    r"([A-I]) is the most appropriate",
    r"([1-9]) is the most appropriate",
    r"I recommend option ([A-I])",
    r"I recommend option ([1-9])",
    r"CORRECT\s+ANSWER\s+IS\s+([A-I])",
    r"CORRECT\s+ANSWER\s+IS\s+([1-9])",
    r"^([A-I])$",
    r"^([A-I])\s*\.$",
    r"^([1-9])$",
    r"^([1-9])\s*\.$",
    r"correct\s+answer\s+is\s+([1-9])",
    r":\n\n([A-I])\)",
    r":\n\n([1-9])\)",
    r":\n([A-I])\.:?",
    r":\n([1-9])\.:?",
    r":\n([A-I])\)",
    r":\n([1-9])\)",
    r"The most appropriate.* ([A-F])\)",
    r"The most appropriate.* ([1-9])\)",

    r"Answer: ([A-F]) is the most appropriate",
    r"Answer: ([1-9]) is the most appropriate",
    r"Answer: ([A-F]),",
    r"Answer: ([1-9]),",
    r"Answer:([A-I])\.:?",
    r"Answer:([1-9])\.:?",
    r"Answer:\n\#([A-I])",
    r"Answer:\n\#([1-9])",
    r"Answer:\n([A-I])\.:?",
    r"Answer:\n([1-9])\.:?",

    r"This statement is correct.*that ([A-I])\.:?",
    r"This statement is correct.*that ([1-9])\.:?",
    r"Answer: \[\/INST\].*:\n\nOption ([A-I])\.:?",
    r"Answer: \[\/INST\].*:\n\nOption ([1-9])\.:?",
    r"Answer: \[\/INST\].*statement \(([A-I])\)",
    r"Answer: \[\/INST\].*statement \(([1-9])\)",
    r"The correct.*is \[([A-I])\]",
    r"The correct.*is \[([1-9])\]",

    r"Answer: \[\/INST\] ([A-I])\)",
    r"Answer: \[\/INST\] ([1-9])\)",
    r"Answer: \[\/INST\] \- ([A-I])",
    r"Answer: \[\/INST\] \- ([1-9])",
    r"Answer: \[\/INST\] ([A-I])\.:?",
    r"Answer: \[\/INST\] ([1-9])\.:?",
    r"Answer: \[\/INST\].*\(([A-I])\)\.:?",
    r"Answer: \[\/INST\].*\(([1-9])\)\.:?",
    r"Answer: \[\/INST\] ([A-I])\n",
    r"Answer: \[\/INST\] ([1-9])\n",
    r"Answer: \[\/INST\] \(([A-I])\)",
    r"Answer: \[\/INST\] \(([1-9])\)",
    r"Answer: \[\/INST\] Option ([A-I]):",
    r"Answer: \[\/INST\] Option ([1-9]):",

    r"Answer:\[\/INST\] ([A-I])\)",
    r"Answer:\[\/INST\] ([1-9])\)",
    r"Answer:\[\/INST\] \- ([A-I])",
    r"Answer:\[\/INST\] \- ([1-9])",
    r"Answer:\[\/INST\] ([A-I])\.:?",
    r"Answer:\[\/INST\] ([1-9])\.:?",
    r"Answer:\[\/INST\].*\(([A-I])\)\.:?",
    r"Answer:\[\/INST\].*\(([1-9])\)\.:?",
    r"Answer:\[\/INST\] ([A-I])\n",
    r"Answer:\[\/INST\] ([1-9])\n",
    r"Answer:\[\/INST\] \(([A-I])\)",
    r"Answer:\[\/INST\] \(([1-9])\)",

    r"Answer:\n\n\- ([A-I])\.:?",
    r"Answer:\n\n\- ([1-9])\.:?",
    r"Answer:\n\- ([A-I])\.:?",
    r"Answer:\n\- ([1-9])\.:?",
    r"Answer:\n\n\- ([A-I])\)",
    r"Answer:\n\n\- ([1-9])\)",
    r"Answer:\n\- ([A-I])\)",
    r"Answer:\n\- ([1-9])\)",
    r"Answer:\n\n\- \[([A-I])\]",
    r"Answer:\n\n\- \[([1-9])\]",
    r"Answer:\n\- \[([A-I])\]",
    r"Answer:\n\- \[([1-9])\]",
    r'Answer: \[([A-I])\]',
    r'Answer: \[([1-9])\]',
    r'Answer: ([A-I]) is correct',
    r'Answer: ([1-9]) is correct',
    r'Answer: ([A-I])  is correct',
    r'Answer: ([1-9])  is correct',
    r'Answer: ([A-I]):',
    r'Answer: ([1-9]):',
    r'Answer: \(([A-I])\)',
    r'Answer: \(([1-9])\)',
    r'Answer: Option ([A-I])',
    r'Answer: Option ([1-9])',
    r'Answer:/nCorrect option is: ([A-I])',
    r'Answer:/nCorrect option is: ([1-9])',

    r'Answer: ([A-I])\)',
    r'Answer: ([A-I])\.:?',
    r'answer: ([1-9])\.:?',
    r'Answer & Explanation: ([A-I])\.:?',
    r'Answer: ([1-9])\.:?',
    r'Answer: ([1-9])\n',
    r'Answer: ([A-I])\n',
    r'Answer: ([1-9])$',
    r'Answer: ([A-I])$',
    r"Answer: '([1-9])\.:?",
    r'Answers: ([1-9])',
    r'Ans: ([1-9])\.:?',
    r'Ans: ([A-I])\.:?',
    r'answer: ([1-9])\.:?',
    r"Answers: \['([1-9])",
    r'answer should be "([1-9])\.:?',
    r"answer is:\n'([1-9])\.:?",
    r"answer is:\n\n'([1-9])\.:?",
    r"answer is:\n'([A-I])\.:?",
    r"answer is:\n\n'([A-I])\.:?",
    r'answer is: "([1-9])\.:?',
    r"answer is ([1-9])",
    r"answer is  ([1-9])",
    r'answer is: "([A-I])',
    r'answer is:  ([A-I])\n',
    r'answer is:  ([1-9])\n',
    r'answer is: ([A-I])\n',
    r'answer is: ([1-9])\n',
    r'answer is "([A-I])',
    r"Choice ([1-9])",
    r"Choice ([A-I])",
    r"choice ([1-9])",
    r"choice ([A-I])",
    r"answer is ([A-I])",
    r"answer is  ([A-I])",
    r"answer is ([a-i])",
    r"answer is  ([a-i])",
    r"^([A-I])$",
    r'^\(([A-I])\)',
    r"^([1-9])$",
    r'^\(([1-9])\)',
    r"Answer: '([A-I])\)",
    r"correct option is ([A-I])\.:?",
    r"correct option is ([1-9])\.:?",
    r"would be option ([A-I]),",
    r"would be option ([1-9]),",
    r"^([A-I])\.:?.*(?!([A-I])\.:?)",
    r"^([1-9])\.:?.*(?!([1-9])\.:?)",
    r'Explanation: ([A-I]) is correct',
    r'Explanation: ([1-9]) is correct',
    r'Answer: Option ([A-I]),',
    r'Answer: Option ([1-9]),',
    r'correct answer is option ([A-I])',
    r'correct answer is option ([1-9])',
    r'correct answer is choice ([A-I])',
    r'correct answer is choice ([1-9])',
    r'correct answer is ([A-I])',
    r'correct answer is ([1-9])',
    r"correct answer is '([A-I])\.:?",
    r"correct answer is '([1-9])\.:?",
    r'answer is \(([A-I])\)',
    r'answer is \(([1-9])\)',
    r"correct answer is ‘([A-I])",
    r"correct answer is ‘([1-9])",
    r'answer is choice \#([A-I])',
    r'answer is choice \#([1-9])',
    r"answer is choice \[([A-I])\]",
    r"answer is choice \[([1-9])\]",
    r'best recommendation.*would be option ([A-I])',
    r'best recommendation.*would be option ([1-9])',
    r'best recommendation.*would be choice ([A-I])',
    r'best recommendation.*would be choice ([1-9])',
    r':\n\n([A-I])\.:?',
    r':\n\n([1-9])\.:?',
    r'most appropriate.*would be option ([A-I])',
    r'most appropriate.*would be option ([1-9])',
    r'most appropriate.*would be Option ([A-I])',
    r'most appropriate.*would be Option ([1-9])',
    r'most appropriate.*would be choice ([A-I])',
    r'most appropriate.*would be choice ([1-9])',
    r'most appropriate.*would be ([A-I])',
    r'most appropriate.*would be ([1-9])',
    r'answer would be ([A-I])',
    r'answer would be ([1-9])',
    r'answer would be \(([A-I])\)',
    r'answer would be \(([1-9])\)',
    r'answer would be option ([A-I])',
    r'answer would be option ([1-9])',
    r'answer would be choice ([A-I])',
    r'answer would be choice ([1-9])',
    r'would advise.*option ([A-I])',
    r'would advise.*option ([1-9])',
    r'would advise.*choice ([A-I])',
    r'would advise.*choice ([1-9])',
    r'most likely diagnosis is ([A-I])',
    r'most likely diagnosis is ([1-9])',
    r'I would choose option ([A-I])',
    r'I would choose option ([1-9])',
    r'I would choose choice ([A-I])',
    r'I would choose choice ([1-9])',
    r'best advice.*would be option ([A-I])',
    r'best advice.*would be option ([1-9])',
    r'best advice.*would be choice ([A-I])',
    r'best advice.*would be choice ([1-9])',
    r'\(Option ([A-I])\) seems to be the most appropriate choice',
    r'\(Option ([1-9])\) seems to be the most appropriate choice',
    r'\(Choice ([A-I])\) seems to be the most appropriate choice',
    r'\(Choice ([1-9])\) seems to be the most appropriate choice',
    r'\(choice ([A-I])\) seems to be the most appropriate choice',
    r'\(choice ([1-9])\) seems to be the most appropriate choice',
    r'ased on the given information.*option ([A-I])',
    r'ased on the given information.*option ([1-9])',
    r'ased on the information.*option ([A-I])',
    r'ased on the information.*option ([1-9])',
    r'ased on.*is ([A-I])',
    r'ased on.*is ([1-9])',
    r'ased on the given information.*choice ([A-I])',
    r'ased on the given information.*choice ([1-9])',
    r'ased on the information.*choice ([A-I])',
    r'ased on the information.*choice ([1-9])',
    r'I would recommend option ([A-I])',
    r'I would recommend option ([1-9])',
    r'I would recommend choice ([A-I])',
    r'I would recommend choice ([1-9])',
    r'best clinical approach.*would be option ([A-I])',
    r'best clinical approach.*would be option ([1-9])',
    r'option ([A-I]).*is the most appropriate choice',
    r'option ([1-9]).*is the most appropriate choice',
    r'option ([A-I]).*is the most appropriate answer',
    r'option ([1-9]).*is the most appropriate answer',
    r'he best.*would be ([A-I])',
    r'he best.*would be ([1-9])',
    r'he best.*would be option ([A-I])',
    r'he best.*would be option ([1-9])',
    r'he best option.*is ([A-I])',
    r'he best option.*is ([1-9])',
    r'he most accurate.*would be option ([A-I])',
    r'he most accurate.*would be option ([1-9])',
    r'he most.*would be ([A-I])',
    r'he most.*would be ([1-9])',
    r'he best medication.*\(Option ([A-I])\)\.:?',
    r'he best medication.*\(Option ([1-9])\)\.:?',
    r'\[\/INST\] ([A-I])\.:?',
    r'\[\/INST\] ([1-9])\.:?',
    r'is option ([A-I])\.:?',
    r'is option ([1-9])\.:?',
    r'is option ([A-I]):',
    r'is option ([1-9]):',
    r'is option ([A-I]),',
    r'is option ([1-9]),',
    r'would be option ([A-I]):',
    r'would be option ([1-9]):',
    r'([A-I])\.:?.* is the most appropriate option',
    r'([1-9])\.:?.* is the most appropriate option',
    r'most likely.*is \(([A-I])\)',
    r'most likely.*is \(([1-9])\)',
    r"([A-I]) is the correct answer",
    r"([1-9]) is the correct answer",
    r"he MOST appropriate.*is ([A-I])\)",
    r"he MOST appropriate.*is ([1-9])\)",
    r"he most appropriate.*is ([A-I])\)",
    r"he most appropriate.*is ([1-9])\)",
    r"he MOST appropriate.*is ([A-I])\.:?",
    r"he MOST appropriate.*is ([1-9])\.:?",
    r"he most appropriate.*is ([A-I])\.:?",
    r"he most appropriate.*is ([1-9])\.:?",
    r"he MOST appropriate.*would be ([A-I])\)",
    r"he MOST appropriate.*would be ([1-9])\)",
    r"he most appropriate.*would be ([A-I])\)",
    r"he most appropriate.*would be ([1-9])\)",
    r"he MOST appropriate.*would be ([A-I])\.:?",
    r"he MOST appropriate.*would be ([1-9])\.:?",
    r"he most appropriate.*would be ([A-I])\.:?",
    r"he most appropriate.*would be ([1-9])\.:?",

    r"he most appropriate.*\(([A-I])\)\.:?",
    r"he most appropriate.*\(([1-9])\)\.:?",

    r"he most appropriate.*is \*\*([A-I])\.:?",
    r"he most appropriate.*is \*\*([1-9])\.:?",

    r"he MOST appropriate.*\(option ([A-I])\)",
    r"he MOST appropriate.*\(option ([1-9])\)",
    r"he most appropriate.*\(option ([A-I])\)",
    r"he most appropriate.*\(option ([1-9])\)",
    r"he MOST appropriate.*\(option ([A-I])\)\.:?",
    r"he MOST appropriate.*\(option ([1-9])\)\.:?",
    r"he most appropriate.*\(option ([A-I])\)\.:?",
    r"he most appropriate.*\(option ([1-9])\)\.:?",

    r"he MOST likely.*is ([A-I])\.:?",
    r"he MOST likely.*is ([1-9])\.:?",
    r"he most likely.*is ([A-I])\.:?",
    r"he most likely.*is ([1-9])\.:?",
    r"most likely.*is: ([A-I])\.:?",
    r"most likely.*is: ([1-9])\.:?",
    r"he correct statement.*is ([A-I])\.:?",
    r"he correct statement.*is ([1-9])\.:?",
    r"would suggest option ([A-I])",
    r"would suggest option ([1-9])",
    r"he answer to.*is ([A-I])\.:?",
    r"he answer to.*is ([1-9])\.:?",
    r"Option ([A-I]) is recommended",
    r"Option ([1-9]) is recommended",

    r"ption ([A-I]) is the best",
    r"ption ([1-9]) is the best",

    r"\<\|assistant\|\>\n([A-I])\.:?",
    r"\<\|assistant\|\>\n([1-9])\.:?",
    r"\<\|assistant\|\>\n([A-I])\)",
    r"\<\|assistant\|\>\n([1-9])\)",
    r"\<\|assistant\|\>\n([A-I])\,",
    r"\<\|assistant\|\>\n([1-9])\,",
    r"\<\|assistant\|\>\n\- ([A-I])\n",
    r"\<\|assistant\|\>\n\- ([1-9])\n",
    r"\<\|assistant\|\>\n\-([A-I])\n",
    r"\<\|assistant\|\>\n\-([1-9])\n",
    r"\<\|assistant\|\>\n\- ([A-I])\.:?",
    r"\<\|assistant\|\>\n\- ([1-9])\.:?",
    r"\<\|assistant\|\>\n\-([A-I])\.:?",
    r"\<\|assistant\|\>\n\-([1-9])\.:?",
    r"\<\|assistant\|\>\n([A-I]) is the correct answer",
    r"\<\|assistant\|\>\n([1-9]) is the correct answer",
    r"answer is: ([A-I])\.:?",
    r"answer is: ([1-9])\.:?",
    r"answer is:  ([A-I])\.:?",
    r"answer is:  ([1-9])\.:?",
    r"([A-I])\.:? is correct",
    r"([1-9])\.:? is correct",
    r"([A-I]) is correct",
    r"([1-9]) is correct",

    r"best.*\(option ([A-I])\)\.:?",
    r"best.*\(option ([1-9])\)\.:?",
    r"best.*\(([A-I])\)\.:?",
    r"best.*\(([1-9])\)\.:?",

    r"ased on.*\(Option ([A-I])\)\.:?",
    r"ased on.*\(Option ([1-9])\)\.:?",
    r"ased on.*\(option ([A-I])\)\.:?",
    r"ased on.*\(option ([1-9])\)\.:?",
    r'ased on.*\"([A-I])\.:?',
    r'ased on.*\"([1-9])\.:?',
    r"ased on.*in option ([A-I])\.:?",
    r"ased on.*in option ([1-9])\.:?",
    r"ased on.*\(([A-I])\)\.:?",
    r"ased on.*\(([1-9])\)\.:?",
    r"ased on.*would be option ([A-I]):",
    r"ased on.*would be option ([1-9]):",

    r"he option that is most.*\(([A-I])\)\.:?",
    r"he option that is most.*\(([1-9])\)\.:?",

    r"option ([A-I]),.*would be the best course",
    r"option ([1-9]),.*would be the best course",
    r"option ([A-I]),.*, is the best",
    r"option ([1-9]),.*, is the best",

    r"he most likely.*is ([A-I])\)",
    r"he most likely.*is ([1-9])\)",
    r"he most likely.*\(([A-I])\)\.:?",
    r"he most likely.*\(([1-9])\)\.:?",
    r"he most likely.*\(option ([A-I])\)\.:?",
    r"he most likely.*\(option ([1-9])\)\.:?",
    r"he most likely.*\(Option ([A-I])\)\.:?",
    r"he most likely.*\(Option ([1-9])\)\.:?",

    r"Answer:\[\/INST\].*option ([A-I]):",
    r"Answer:\[\/INST\].*option ([1-9]):",
    r"The patient has.*\(Option ([A-I])\)\.:?",
    r"The patient has.*\(Option ([1-9])\)\.:?",
    r"The target.*is ([A-I])\)",
    r"The target.*is ([1-9])\)",

    r"^([a-i])\.:?",
    r"^([a-i])\n",
    r"^ ([a-i])\.:?",
    r"^ ([a-i])\n",
    r"^  ([a-i])\n",
    r"^([A-I])\.:?",
    r"^([1-9])\.:?",
    r"^([A-I])\n",
    r"^([1-9])\n",
    r"^ ([A-I])\.:?",
    r"^ ([1-9])\.:?",
    r"^ ([A-I])\n",
    r"^ ([1-9])\n",
    r"^  ([A-I])\n",
    r"^  ([1-9])\n",
    r"Answer:.*\(([A-I])\)\n",
    r"Answer:.*\(([1-9])\)\n",

    r"suggest.*\(Option ([A-I])\)\.:?",
    r"suggest.*\(Option ([1-9])\)\.:?",

    r"\(([A-I])\) should be seriously considered\.:?",
    r"\(([1-9])\) should be seriously considered\.:?",
    r"he most correct.*\(statement ([A-I])\)\.:?",
    r"he most correct.*\(statement ([1-9])\)\.:?",
    r"he most correct.*\(option ([A-I])\)\.:?",
    r"he most correct.*\(option ([1-9])\)\.:?",
    r"he correct answer is.*\(([A-I])\)\.:?",
    r"he correct answer is.*\(([1-9])\)\.:?",
    r"I would advise.*\(Option ([A-I])\)\.:?",
    r"I would advise.*\(option ([A-I])\)\.:?",
    r"I would advise.*\(Option ([1-9])\)\.:?",
    r"I would advise.*\(option ([1-9])\)\.:?",
    r"Based on.*would be ([A-I])\.:?",
    r"Based on.*would be ([1-9])\.:?",
    r"\(option ([A-I])\) would be the most appropriate",
    r"\(option ([1-9])\) would be the most appropriate",
    r"ased on.*would likely be ([A-I])\.:?",
    r"ased on.*would likely be ([1-9])\.:?",

    r"\(option ([A-I])\) is the intervention with the greatest potential",
    r"A potential risk factor.*in the question is.*\(option ([A-I])\)\.:?",
    r"he preferred.*therapy.*is ([A-I])\.:?",
    r"The criterion that would most favor.*is ([A-I])\.:?",

    r"^\n([A-I])\n",
    r"^\n([1-9])\n",

    r"Answer:\*\* ([A-I])",
    r"Answer:\*\* ([1-9])",

    r"best recommendation.*outlined in option ([A-I])\.:?",
    r"best recommendation.*outlined in option ([1-9])\.:?",
    r"best recommendation.*outlined in Option ([A-I])\.:?",
    r"best recommendation.*outlined in Option ([1-9])\.:?",

    r"([A-I])\) is the best option",
    r"([1-9])\) is the best option",
]
