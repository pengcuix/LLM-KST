import sys
import json
import os
import argparse
import numpy as np
import pandas as pd
from data import config
from dataclasses import dataclass, asdict
from PIL import Image
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from prettytable import PrettyTable
# from data.plot import plot_histogram
from skills.dependency import PoSet

EPS = 1e-8


@dataclass
class Domain:
    domain_id: str
    name: str
    grade: int


@dataclass
class Cluster:
    cluster_id: str
    domain_id: str
    name: str
    concepts: list


@dataclass
class Concept:
    concept_id: str
    cluster_id: str
    description: str
    example_problem: str
    pre_concepts: list
    post_concepts: list
    related_concepts: list


@dataclass
class Question:
    id: str
    type: str
    content: str
    concepts: list
    answer: str
    options: dict
    solution: str
    content_image: []
    option_image: {}


@dataclass
class Response:
    subject_id: str  # model or student
    question_id: str
    label: int  # 1 correct, 0 incorrect
    solution: str  # reasoning steps, only for LLMs
    timestamp: float  # only for students


class NYTMathStandards:
    def __init__(self, file):
        with open(file, 'r') as fp:
            concepts = json.load(fp)   # domain, cluster, standard, edges, nd_edges

        self.domains = {}
        self.clusters = {}
        self.concepts = {}

        for domain_id, v in concepts['domains'].items():
            self.domains[domain_id] = Domain(domain_id=domain_id, name=v['name'], grade=v['grade'])
        for cluster_id, v in concepts['clusters'].items():
            self.clusters[cluster_id] = Cluster(cluster_id=cluster_id, name=v['name'], domain_id=v['ccmathdomain_id'], concepts=[])
        for concept_id, v in concepts['standards'].items():
            self.concepts[concept_id] = Concept(
                concept_id=concept_id,
                cluster_id=v['ccmathcluster_id'],
                description=v['desc'],
                example_problem=v['example_problem'],
                pre_concepts=v['rev_edges'],
                post_concepts=v['edges'],
                related_concepts=v['nd_edges']
            )
            self.clusters[v['ccmathcluster_id']].concepts.append(concept_id)
        # print('here, all concept_ids', sorted([int(x) for x in self.concepts.keys()]))
        self.poset = PoSet(query_fn=None)
        self.poset.nodes = set(self.concepts.keys())
        self.poset.children = {concept_id: set(concept.post_concepts) for concept_id, concept in self.concepts.items()}  # post_requisite
        self.poset.parents = {concept_id: set(concept.pre_concepts) for concept_id, concept in self.concepts.items()}  # pre_requisite
        # print('here parents', self.poset.parents.keys())

    def get_all_prerequisite(self, concept_id, visited=None):
        if visited is None:
            visited = set()
        for pre_id in self.concepts[concept_id].pre_concepts:
            if pre_id not in visited:
                visited.add(pre_id)
                self.get_all_prerequisite(pre_id, visited)
        return visited

    def get_all_postrequisite(self, concept_id, visited=None):
        if visited is None:
            visited = set()
        for post_id in self.concepts[concept_id].post_concepts:
            if post_id not in visited:
                visited.add(post_id)
                self.get_all_postrequisite(post_id, visited)
        return visited

    def get_stats(self):
        table = PrettyTable()
        table.field_names = [
            '# Domains.',
            '# Clusters.',
            '# Concepts.',
            '# avg. prerequisite',
            '# avg. postrequisite',
            '# avg. related'
        ]

        # print('clusters', [cluster.name for cluster in self.clusters.values()])
        # exit(1)
        domains = set([domain.name for domain in self.domains.values()])
        clusters = set([cluster.name for cluster in self.clusters.values()])
        concepts = set([concept.description for concept in self.concepts.values()])
        num_prerequisite = [len(concept.pre_concepts) for concept in self.concepts.values()]
        num_postrequisite = [len(concept.post_concepts) for concept in self.concepts.values()]

        # print('num prerequisite', num_prerequisite, sum(num_prerequisite))
        # print('num postrequisite', num_postrequisite, sum(num_postrequisite))
        # exit(1)
        num_related = [len(concept.related_concepts) for concept in self.concepts.values()]
        table.title = 'XES All Statistics'
        table.add_row([
            len(domains),
            len(clusters),
            len(concepts),
            round(sum(num_prerequisite) / len(num_prerequisite), 2),
            round(sum(num_postrequisite)/len(num_postrequisite), 2),
            round(sum(num_related)/len(num_related), 2)
        ])

        print(table)


class XESDataset:
    def __init__(self, data_dir, lang='en', ignore='image'):
        assert ignore in ['image', 'text', None]
        self.questions = {}
        if lang == 'en':
            # with open(os.path.join(data_dir, 'metadata', 'questions_en.json'), 'r') as fp:
            #     data_raw = json.load(fp)
            # for question_id in tqdm(data_raw, desc='Reading XES questions in EN'):
            #     # dataset filtering
            #     # low quality questions
            #     if int(data_raw[question_id]['valid']) == 0:
            #         continue
            #     if ignore == 'image' and any([data_raw[question_id]['content_image'], data_raw[question_id]['option_image']]):
            #         continue
            #     if ignore == 'text' and not any([data_raw[question_id]['content_image'], data_raw[question_id]['option_image']]):
            #         continue
            #     self.questions[question_id] = Question(
            #         id=question_id,
            #         type=data_raw[question_id]['type'],
            #         content=data_raw[question_id]['content'],
            #         concepts=data_raw[question_id]['kc_routes'],
            #         answer=data_raw[question_id]['answer'],
            #         options=data_raw[question_id]['options'],
            #         solution=data_raw[question_id]['analysis'],
            #         content_image=data_raw[question_id]['content_image'],
            #         option_image=data_raw[question_id]['option_image'],
            #     )

            with open(os.path.join(data_dir, 'questions_en_dedup.jsonl'), 'r') as fp:
                for line in fp.readlines():
                    data = json.loads(line.strip())
                    self.questions[data['orig_idx']] = Question(
                        id=data['orig_idx'],
                        type=data['type'],
                        content=data['question'],
                        concepts=data['kc_routes'],
                        answer=data['answer'],
                        options=data['options'],
                        solution=data['analysis'],
                        content_image=data['question_img'],
                        option_image=data['option_img'],
                    )

        # elif lang == 'cn':
        #     with open(os.path.join(data_dir, 'metadata', 'questions.json'), 'r') as fp:
        #         data_raw = json.load(fp)
        #     for question_id in tqdm(data_raw, desc='Reading XES questions in CN'):
        #         if 'question_' in data_raw[question_id]['content']:
        #             content = data_raw[question_id]['content'].split('question_id')[0].strip()
        #             imgs = data_raw[question_id]['content'].split('question_id')[1:]
        #             content_image = [img.split('image_')[1] for img in imgs]
        #         else:
        #             content = data_raw[question_id]['content']
        #             content_image = []
        #
        #         options = data_raw[question_id]['options']
        #         option_image = {}
        #         for key in options:
        #             if 'question_' in options[key]:
        #                 _ = options[key].split('question_').strip()
        #                 options[key] = _[0]
        #                 option_image[key] = [img.split('image_') for img in _[1]]
        #             else:
        #                 option_image[key] = []
        #
        #         if ignore == 'figure' and any([content_image, option_image]):
        #             continue
        #         if ignore == 'text' and not any([content_image, option_image]):
        #             continue
        #
        #         self.questions[question_id] = Question(
        #             id=question_id,
        #             type=data_raw[question_id]['type'],
        #             content=content,
        #             concepts=[route.split('----') for route in data_raw[question_id]['kc_routes']],
        #             answer=data_raw[question_id]['answer'],
        #             options=options,
        #             solution=data_raw[question_id]['analysis_xes'],
        #             content_image=content_image,
        #             option_image=option_image,
        #         )

        self.interactions = {}
        self.question_student_index = {}

        for file in ['train_valid_sequences_quelevel.csv', 'test_quelevel.csv']:
            df = pd.read_csv(os.path.join(data_dir, 'question_level', file))
            for _, row in tqdm(df.iterrows(), total=len(df), desc='Reading XES interactions from {}'.format(file)):
                if row['uid'] not in self.interactions:
                    self.interactions[row['uid']] = {}

                questions = row['questions'].split(',')
                responses = row['responses'].split(',')
                timestamps = row['timestamps'].split(',')
                if 'selectmask' in row:
                    masks = row['selectmask'].split(',')
                else:
                    masks = [1] * len(questions)
                for i, q_id in enumerate(row['questions'].split(',')):
                    if q_id not in self.questions:
                        continue
                    if int(masks[i]) == -1:
                        break
                    self.interactions[row['uid']][q_id] = Response(
                        subject_id=row['uid'],
                        question_id=questions[i],
                        label=int(responses[i]),  # 1 correct, 0 incorrect
                        solution='',  # reasoning steps, only for LLMs
                        timestamp=float(timestamps[i])  # only for students
                    )

                if not self.interactions[row['uid']]:
                    self.interactions.pop(row['uid'])

        for stu_id in self.interactions:
            for q_id in self.interactions[stu_id]:
                if q_id not in self.question_student_index:
                    self.question_student_index[q_id] = set()
                self.question_student_index[q_id].add(stu_id)

    def get_stats(self):
        # num students, num questions, interaction (median, min, max), stud_performance (mean, min, max),
        table = PrettyTable()
        table.field_names = [
            '# Stu.',
            '# Ques.',
            '# Inter.',
            'Inter./Stu (Mean/Median/Min/Max)',
            'Performance (Mean/Median/Min/Max)'
        ]

        num_students = len(self.interactions)
        num_questions = len(self.questions)

        inter_cnt = [len(stu_inters) for stu_inters in self.interactions.values()]

        score = [np.mean([inter.label for inter in stu_inters.values()]) for stu_inters in self.interactions.values()]
        table.title = 'XES All Statistics'
        table.add_row([
            num_students,
            num_questions,
            np.sum(inter_cnt),
            f"{np.mean(inter_cnt):.2f}/{np.median(inter_cnt):.2f}/{np.min(inter_cnt):.2f}/{np.max(inter_cnt):.2f}",
            f'{np.mean(score):.2f}/{np.median(score):.2f}/{np.min(score):.2f}/{np.max(score):.2f}'
        ])

        print(table)

    def plot_stu_ability(self):
        theta = []
        for stu_id in self.interactions:
            acc = sum([item.label for item in self.interactions[stu_id].values()]) / len(self.interactions[stu_id])
            theta.append(acc)
        plot_histogram(theta, bins=np.arange(0, 1.05, 0.05), x_label='Accuracy', y_label='Count')

    def plot_item_difficulty(self):
        q_diff = {}
        for stu_id in self.interactions:
            for q_id in self.interactions[stu_id]:
                if q_id not in q_diff:
                    q_diff[q_id] = [0, 0]
                q_diff[q_id][self.interactions[stu_id][q_id].label] += 1

        for q_id in q_diff:
            q_diff[q_id] = q_diff[q_id][0] / (q_diff[q_id][1] + q_diff[q_id][0])

        diff = list(q_diff.values())
        plot_histogram(data=diff, bins=np.arange(0, 1.05, 0.05), x_label='Difficulty (Error rate)', y_label='Count')

    def get_question_stats(self):
        question_stats = {q_id: {'n_correct': 0, 'n_wrong': 0, 'n_total': 0, 'error_rate': 0.} for q_id in self.questions}

        for stu_id in self.interactions:
            for q_id in self.interactions[stu_id]:
                if q_id not in question_stats:
                    question_stats[q_id] = {}
                question_stats[q_id]['n_total'] += 1
                if self.interactions[stu_id][q_id].label == 1:
                    question_stats[q_id]['n_correct'] += 1
                else:
                    question_stats[q_id]['n_wrong'] += 1

        for q_id in question_stats:
            if question_stats[q_id]['n_total'] == 0:
                continue
            question_stats[q_id]['error_rate'] = question_stats[q_id]['n_wrong'] / question_stats[q_id]['n_total']

        return question_stats

    def get_conditional_question_stats(self, cached):
        if os.path.exists(cached):
            with open(cached, 'r') as fp:
                return json.load(fp)
        # cqs[q_i][q_j]: P(q_i=0|q_j=1)
        conditional_question_stats = {q_i: {q_j: {'n_correct': 0, 'n_wrong': 0, 'n_total': 0, 'error_rate': 0.} for q_j in self.questions} for q_i in self.questions}
        for stu_id in tqdm(self.interactions, desc="Calculating P(A|B)"):
            stu_data = self.interactions[stu_id]
            # 预先拆分（关键优化）
            correct_qs = [q for q, v in stu_data.items() if v.label == 1]
            wrong_qs = [q for q, v in stu_data.items() if v.label == 0]
            # 只从 q_j = correct 出发
            for q_j in correct_qs:
                # q_i wrong
                for q_i in wrong_qs:
                    conditional_question_stats[q_i][q_j]['n_total'] += 1
                    conditional_question_stats[q_i][q_j]['n_wrong'] += 1
                # q_i correct（排除自己）
                for q_i in correct_qs:
                    if q_i == q_j:
                        continue
                    conditional_question_stats[q_i][q_j]['n_total'] += 1
                    conditional_question_stats[q_i][q_j]['n_correct'] += 1

        for q_i in conditional_question_stats:
            for q_j in conditional_question_stats[q_i]:
                conditional_question_stats[q_i][q_j]['error_rate'] = conditional_question_stats[q_i][q_j]['n_wrong'] / (conditional_question_stats[q_i][q_j]['n_total'] + 1e-5)

        with open(cached, 'w') as fp:
            json.dump(conditional_question_stats, fp)

        return conditional_question_stats

    def plot_interaction_time(self):
        pass

    def cluster_students_by_performance(self):
        pass


class NipsEduDataset:
    def __init__(self, data_dir):
        self.questions = {}
        self.interactions = {}

        with open(os.path.join(data_dir, 'questions.jsonl'), 'r') as fp:
            for line in fp.readlines():
                question = json.loads(line)
                self.questions[question['id']] = Question(
                    id=question['id'],
                    type='multiple-choice',
                    content=question['options'],
                    concepts=[],
                    answer='',  # TODO
                    options=question['options'],
                    solution='',
                    content_image=[],
                    option_image={}
                )

        df = pd.read_csv(os.path.join(data_dir, 'train', 'train_task_3_4.csv'))

    def get_stats(self):
        pass

