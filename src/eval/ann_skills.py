import argparse
import logging
import os
import sys
import numpy as np
import json
from pathlib import Path
from data.data_utils import NipsEduDataset, XESDataset, NYTMathStandards, Question
from skills.extract import extract_skills, extract_skills_reference
from skills.cluster import cluster_labeling, skill_clustering
from skills.dependency import build_poset
from data import config


def annotate_free(dataset):
    problems = {problem.id: {'content': problem.content, 'solution': problem.solution, 'reference_skills': None} for problem in dataset.questions.values()}
    with open(args.output_path, 'r') as fp_output:
        skills = json.load(fp_output)  # {'skills_raw', 'q_matrix', 'cluster_labels', 'skills_clustered', 'poset', 'dep_annotations}

    # extract skills
    if 'skills_raw' not in skills or 'q_matrix' not in skills:
        skills_raw, q_matrix = extract_skills(problems=problems)
        skills['skills_raw'] = skills_raw
        skills['q_matrix'] = q_matrix

    print(f'{len(problems)} problems, {len(skills["skills_raw"])} skills')

    # cluster + labeling
    if 'cluster_labels' not in skills or 'skills_clustered' not in skills:
        clustered_skills = skills['skills_raw']
        cluster_labels = [0 for i in range(len(clustered_skills))]
        prev_size = len(clustered_skills)
        cur_size = 0
        iter_id = 1
        while prev_size > cur_size and iter_id < 4:
            prev_size = len(clustered_skills)
            cluster_labels, _ = skill_clustering(skill_set=clustered_skills, distance_threshold=args.clustering_distance_threshold)
            cur_size = max(cluster_labels) + 1
            print('Clustering iteration {}, {} skills -> {} skills'.format(iter_id, prev_size, cur_size))
            clustered_skills = cluster_labeling(clustered_skills, cluster_labels)
            iter_id += 1
        skills['skills_clustered'] = clustered_skills
        skills['cluster_labels'] = cluster_labels

    # dependency
    if 'deps' not in skills:
        poset, deps = build_poset(skills['skills_clustered'])
        skills['prerequisite'], skills['deps'] = poset.parents, deps



def annotate_reference(questions, max_cnt=-1):
    nyt_standards = NYTMathStandards(config.NYS_MATH_STANDARDS)
    skill_clusters = [f'{cluster.cluster_id}: {cluster.name}' for cluster in nyt_standards.clusters.values()]
    problems = {}
    for idx, problem in enumerate(questions.values()):
        if idx >= max_cnt > 0:
            break
        problems[problem.id] = {'content': problem.content, 'solution': problem.solution, 'reference_skills': '\n'.join(skill_clusters)}

    print('{} domains, {} clusters, {} skills'.format(len(nyt_standards.domains), len(nyt_standards.clusters), len(nyt_standards.concepts)))
    # for domain_id in nyt_standards.domains:
    #     print(domain_id, nyt_standards.domains[domain_id])
    # for concept_id in nyt_standards.concepts:
    #     print(concept_id, nyt_standards.concepts[concept_id].description)

    logging.info('Annotating clusters')
    _, question2cluster = extract_skills_reference(problems=problems)

    logging.info('Annotating concepts')
    skip_list = set()
    for problem_id in question2cluster:
        if not question2cluster[problem_id]:
            skip_list.add(problem_id)
        concepts = []
        for cluster_id in question2cluster[problem_id]:
            cluster_id = str(cluster_id)
            if cluster_id not in nyt_standards.clusters:
                print('not found cluster', cluster_id)
                continue
            for concept_id in nyt_standards.clusters[cluster_id].concepts:
                concepts.append((concept_id, nyt_standards.concepts[concept_id].description))
        reference_skills = '\n'.join(['{}: {}'.format(c_id, c_desc) for c_id, c_desc in concepts])
        problems[problem_id]['reference_skills'] = reference_skills

    _, q_matrix = extract_skills_reference(problems, skip_list=skip_list)

    print(q_matrix)

    for problem_id in skip_list:
        q_matrix[problem_id] = []

    with open('xes_skills_nys.json', 'w') as fp:
        json.dump(q_matrix, fp)

    return q_matrix


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nys_examples')
    parser.add_argument('--output_path', type=Path, default='output/skills/xes_skills.json')
    parser.add_argument('--clustering_distance_threshold', type=float, default=0.3)
    parser.add_argument('--pipeline', type=lambda x: x.split(','), default='extract,cluster,label,')
    parser.add_argument('--max_clustering_iteration', type=int, default=5)

    args = parser.parse_args()

    print('Loading {} dataset'.format(args.dataset))
    if args.dataset == 'nips_edu':
        data = NipsEduDataset(data_dir=config.NIPS_EDU_DIR)
    elif args.dataset == 'xes':
        data = XESDataset(data_dir=config.XES3G5M_DIR)
        questions = data.questions
    elif args.dataset == 'nys_examples':
        with open('example_questions.json', 'r') as fp:
            data = json.load(fp)
        questions = {concept_id:
            Question(
                id=concept_id,
                type=None,
                content=data[concept_id],
                concepts=concept_id,
                answer='No',
                options=None,
                solution='No',
                content_image=None,
                option_image=None
            )
            for concept_id in data
        }


    # annotate_example
    annotations = annotate_reference(questions, max_cnt=-1)

    with open('example_annotation_multi.json', 'w') as fp:
        json.dump(annotations, fp)




