""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function

import json
import re
import string
import sys
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    import random
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                answers = qa["answers"]
                # Note only one answer
                answer_text = answers[random.choice(range(len(answers)))]["text"]
                answer_start = answers[random.choice(range(len(answers)))]["answer_start"]
                answer_end = answer_start + len(answer_text) -1
                #ground_truths = list(map(lambda x: x['text'], qa['answers']))
                # prediction is (start, end)
                pred_start, pred_end = predictions[qa['id']]
                left = min(answer_start, pred_start)
                right = max(answer_end, pred_end)
                span = right - left + 1
                left_residual = max(answer_start, pred_start) - left
                right_residual = right -  min(answer_end, pred_end)
                overlap = span - left_residual - right_residual
                if overlap > 0:
                    precision = float(overlap) / (pred_end - pred_start + 1)
                    recall = float(overlap) / (answer_end - answer_start + 1)
                else:
                    precision = 0.
                    recall = 0.
                assert precision >= 0 and recall >= 0 
                
                f1 += (2 * precision * recall) / (precision + recall) if (overlap > 0.) else 0
                
                #exact_match += metric_max_over_ground_truths(
                #    exact_match_score, prediction, ground_truths)
                #f1 += metric_max_over_ground_truths(
                #    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def evaluate_from_file(dataset_file, prediction_file):
    expected_version = '1.1'
    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    return evaluate(dataset, predictions)
