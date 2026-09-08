import random
import sys
import json
import os
import argparse
import numpy as np
from data.data_utils import XESDataset, NipsEduDataset
from data import config
from data.data_utils import Response, EPS
from tqdm import tqdm
from pathlib import Path


def eval_acc(responses):
    subject_acc = {}
    for sub_id in tqdm(responses):
        num_correct = 0
        num_wrong = 0
        for q_id in responses[sub_id]:
            if responses[sub_id][q_id].label == 1:
                num_correct += 1
            else:
                num_wrong += 1
        subject_acc[sub_id] = num_correct / (num_correct + num_wrong)
    return subject_acc


def eval_psr_subject(question_prerequisite, sub_responses, level='macro'):
    assert level in ['micro', 'macro']

    subject_psr = {}

    for sub_id in tqdm(sub_responses):

        mastered_questions = set()
        for q_id in sub_responses[sub_id]:
            if sub_responses[sub_id][q_id].label == 1:
                mastered_questions.add(q_id)

        if level == 'micro':
            # edge-level aggregation
            total_correct_pre = 0
            total_pre = 0

            for q_id in mastered_questions:
                if q_id not in question_prerequisite:
                    continue
                pre_reqs = question_prerequisite[q_id]
                for pre_q in pre_reqs:
                    if pre_q not in sub_responses[sub_id]:
                        continue
                    total_pre += 1
                    if sub_responses[sub_id][pre_q].label == 1:
                        total_correct_pre += 1
            if total_pre == 0:
                continue
            psr = total_correct_pre / total_pre

        else:  # macro
            psr_list = []
            for q_id in mastered_questions:
                if q_id not in question_prerequisite:
                    continue

                prereqs = question_prerequisite[q_id]
                valid_pre = [p for p in prereqs if p in sub_responses[sub_id]]

                if len(valid_pre) == 0:
                    continue

                num_pre_correct = sum(
                    sub_responses[sub_id][p].label == 1 for p in valid_pre
                )

                psr_q = num_pre_correct / len(valid_pre)
                psr_list.append(psr_q)

            if len(psr_list) == 0:
                continue
            psr = np.mean(psr_list)

        subject_psr[sub_id] = psr

    return subject_psr


def eval_psr_item(question_prerequisite, sub_responses):
    item_psr = {q_id: {'num_total_pre': 0, 'num_correct_pre': 0, 'psr': 0.} for q_id in question_prerequisite}
    # item_acc = {q_id: {'num_correct': 0, 'num_wrong': 0} for q_id in question_prerequisite}
    for q_id in tqdm(question_prerequisite, desc='Item PSR'):
        for sub_id in sub_responses:
            # skip if not observed
            if q_id not in sub_responses[sub_id]:
                continue
            # compute acc
            # if sub_responses[sub_id][q_id].label == 1:
            #     item_acc[q_id]['num_correct'] += 1
            # else:
            #     item_acc[q_id]['num_wrong'] += 1
            #     continue

            # only compute psr when q is correct
            if sub_responses[sub_id][q_id].label == 0:
                continue
            # compute psr
            for pre_q_id in question_prerequisite[q_id]:
                if pre_q_id in sub_responses[sub_id]:
                    item_psr[q_id]['num_total_pre'] += 1
                    if sub_responses[sub_id][pre_q_id].label == 1:
                        item_psr[q_id]['num_correct_pre'] += 1

    for q_id in item_psr:
        item_psr[q_id]['psr'] = item_psr[q_id]['num_correct_pre'] / (item_psr[q_id]['num_total_pre'] + 1e-5)

    # if sub_id not in dist:
    #     dist[sub_id] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    psr_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for sub_id in sub_responses:
        for q_i in sub_responses[sub_id]:
            correct_cnt = 0
            incorrect_cnt = 0
            if sub_responses[sub_id][q_i].label == 0:
                continue
            if q_i not in question_prerequisite:
                continue
            for q_j in question_prerequisite[q_i]:
                if q_j not in sub_responses[sub_id]:
                    continue
                if sub_responses[sub_id][q_j].label == 1:
                    correct_cnt += 1
                else:
                    incorrect_cnt += 1
            if correct_cnt + incorrect_cnt >= 5:
                psr = correct_cnt / (correct_cnt + incorrect_cnt)
                if psr == 1:
                    bucket = 5
                else:
                    bucket = int((psr+1e-9) // 0.2)
                psr_dist[bucket] += 1

    total = sum(psr_dist.values())
    for key in psr_dist:
        psr_dist[key] /= total

    print(f'PSR dist: [0-0.2)={psr_dist[0]} | [0.2, 0.4)={psr_dist[1]} | [0.4, 0.6)={psr_dist[2]} | [0.6-0.8)={psr_dist[3]} | [0.8-1.0)={psr_dist[4]} | 1.0={psr_dist[5]}')

    return item_psr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='xes')
    parser.add_argument('--target', type=str)  # subject, item
    parser.add_argument('--q_type', type=str, default='all')
    # 'Llama-3.1-8B-Instruct' 'Llama-3.1-70B-Instruct' 'Mistral-7B-Instruct-v0.3' 'Qwen2.5-32B-Instruct' 'Qwen2.5-7B-Instruct' 'claude-sonnet-4-6' 'gpt-4.1-mini'
    parser.add_argument('--subject', type=str, default='human')
    parser.add_argument('--output_path', type=Path)
    parser.add_argument('--deps_grounding', type=str, default='skill')
    args = parser.parse_args()

    print('Args dataset: {}, q_type: {}, subject: {}'.format(args.dataset, args.q_type, args.subject))

    # subject responses
    if args.subject == 'human':
        if args.dataset == 'xes':
            dataset = XESDataset(config.XES3G5M_DIR)
        else:
            dataset = NipsEduDataset(config.NIPS_EDU_DIR)
        subject_responses = dataset.interactions

    else:
        subject_responses = {args.subject: {}}
        for filepath in config.OUTPUT[args.dataset][args.subject][args.q_type]:
            with open(filepath, 'r') as fp:
                for line in fp.readlines():
                    data = json.loads(line.strip())
                    subject_responses[args.subject][data['orig_idx']] = Response(
                        subject_id=args.subject,
                        question_id=data['orig_idx'],
                        label=int(data['score'][0]),
                        solution=data['code'],
                        timestamp=0.
                    )

    print('{} subjects, {} questions'.format(len(subject_responses), sum([len(q_ids) for q_ids in subject_responses.values()])))

    # prerequisite questions
    with open('data/dataset/xes_pre_questions_skill_verified_sampled.json', 'r') as fp_pre_questions:
        pre_questions = json.load(fp_pre_questions)

    keys = list(pre_questions.keys())
    # sampled_keys = random.sample(keys, k=int(len(keys) * 0.30))
    # pre_questions = {key: pre_questions[key] for key in sampled_keys}
    # print('here', len(pre_questions))
    # exit(1)
    if args.target == 'subjects':
        acc_result = eval_acc(subject_responses)
        psr_result_macro = eval_psr_subject(
            question_prerequisite=pre_questions,
            sub_responses=subject_responses,
            level='macro'
        )
        psr_result_micro = eval_psr_subject(
            question_prerequisite=pre_questions,
            sub_responses=subject_responses,
            level='micro'
        )
        # with open(args.output_path, 'w') as fp_output:
        #     json.dump({'subject_id': args.subject, 'subject_acc': acc_result, 'subject_psr_macro': psr_result_macro, 'subject_psr_micro': psr_result_micro}, fp_output)

        print(f'{args.subject}, acc={np.mean(list(acc_result.values()))}')
        print(f'{args.subject}, psr_micro={np.mean(list(psr_result_micro.values()))}')
        print(f'{args.subject}, psr_macro={np.mean(list(psr_result_macro.values()))}')

    elif args.target == 'items':
        print('='*100)
        print(args.subject)
        item_psr_result = eval_psr_item(question_prerequisite=pre_questions, sub_responses=subject_responses)
        # all_psrs = [v['psr'] for k, v in item_psr_result.items() if v['num_total_pre'] > 5]

        # x = np.array(all_psrs)
        # bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # counts, edges = np.histogram(x[x < 1], bins=bins)
        #
        # # 比例
        # ratios = counts / len(x)
        # # 单独统计 ==1
        # ratio_eq_1 = np.mean(x == 1)
        #
        # print("0-0.2:", ratios[0])
        # print("0.2-0.4:", ratios[1])
        # print("0.4-0.6:", ratios[2])
        # print("0.6-0.8:", ratios[3])
        # print("0.8-1.0:", ratios[4])
        # print("=1:", ratio_eq_1)
        # with open(args.output_path, 'w') as fp_output:
        #     json.dump({'subject_id': args.subject, 'item_psr': item_psr_result}, fp_output)
