import os
from pathlib import Path

XES3G5M_DIR = Path('data/dataset/XES3G5M')
NIPS_EDU_DIR = Path('data/dataset/NeurIPS_Edu')
NYS_MATH_STANDARDS = Path('data/dataset/nys_concepts.json')


openai_api_key = os.getenv("OPENAI_API_KEY_LLM_knowledge_dep")

api_cost = {
    'gpt-4o-mini': {'input': 0.15/1000000, 'output': 0.60/1000000},
    'gpt-4.1-mini': {'input': 0.4/1000000, 'output': 1.60/1000000}
}

'''
skills: skills_raw, skills_clustered, cluster_labels, q_martix, skill_dep, dep_annotations,  
'''

skill_ann_path = {
    'xes': {
        'skills_free':  "output/skills/xes_skills_free.json",
        'skills_free_filtered':  "output/skills/xes_skills_free_filtered.json",
        'skills_nys': 'output/skills/xes_skills_nys.json',
        'skills_nys_filtered':  "output/skills/xes_skills_nys_filtered.json",
        'pre_questions_pattern': 'output/skills/xes_pre_questions_pattern.json',
        'pre_questions_pattern_filtered': 'output/skills/xes_pre_questions_pattern_filtered.json',
        'pre_questions_skill_raw': 'data/dataset/xes_pre_questions_skill_raw.json',
        'pre_questions_skill_filtered': 'data/dataset/xes_pre_questions_skill_filtered_v2.json'
    },
    'nips': "output/skills/nips_skills.json",
}

OUTPUT = {
    'xes': {
        'Llama-3.1-8B-Instruct': {
            'fib': ['llm_running/math_evaluation/results_xes/Llama-3.1-8B-Instruct-seed42/xes/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Llama-3.1-8B-Instruct-seed42/xes_opt/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Llama-3.1-8B-Instruct-seed42/xes/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Llama-3.1-8B-Instruct-seed42/xes_opt/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'
            ],
        },
        'Llama-3.1-70B-Instruct': {
            'fib': ['llm_running/math_evaluation/results_xes/Llama-3.1-70B-Instruct-seed42/xes/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Llama-3.1-70B-Instruct-seed42/xes_opt/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Llama-3.1-70B-Instruct-seed42/xes/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Llama-3.1-70B-Instruct-seed42/xes_opt/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'
            ]
        },
        'Mistral-7B-Instruct-v0.3': {
            'fib': ['llm_running/math_evaluation/results_xes/Mistral-7B-Instruct-v0.3-seed42/xes/test_mistral_-1_seed42_t0.6_s0_e-1.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Mistral-7B-Instruct-v0.3-seed42/xes_opt/test_mistral_-1_seed42_t0.6_s0_e-1.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Mistral-7B-Instruct-v0.3-seed42/xes/test_mistral_-1_seed42_t0.6_s0_e-1.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Mistral-7B-Instruct-v0.3-seed42/xes_opt/test_mistral_-1_seed42_t0.6_s0_e-1.jsonl'
            ]
        },
        # 'qwen2.5-7B-Instruct': {
        #     'fib': ['llm_running/math_evaluation/results_xes/Qwen2.5-7B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
        #     'mcq': [''],
        #     'all': []
        # },
        'Qwen2.5-32B-Instruct': {
            'fib': ['llm_running/math_evaluation/results_xes/Qwen2.5-32B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Qwen2.5-32B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Qwen2.5-32B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Qwen2.5-32B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'
            ]
        },
        'Qwen2.5-7B-Instruct': {
            'fib': ['llm_running/math_evaluation/results_xes/Qwen2.5-7B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Qwen2.5-7B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Qwen2.5-7B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Qwen2.5-7B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'
            ]
        },
        'Qwen3-Next-80B-A3B-Instruct': {
            'fib': ['llm_running/math_evaluation/results_xes/Qwen3-Next-80B-A3B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Qwen3-Next-80B-A3B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Qwen3-Next-80B-A3B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Qwen3-Next-80B-A3B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1.jsonl'
            ]
        },
        'claude-sonnet-4-6': {
            'fib': ['llm_running/math_evaluation/results_xes/claude-sonnet-4-6-seed42/xes/test_cot_-1_seed42_t0.0_s0_e-1.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/claude-sonnet-4-6-seed42/xes_opt/test_cot_-1_seed42_t0.0_s0_e-1.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/claude-sonnet-4-6-seed42/xes/test_cot_-1_seed42_t0.0_s0_e-1.jsonl',
                'llm_running/math_evaluation/results_xes_opt/claude-sonnet-4-6-seed42/xes_opt/test_cot_-1_seed42_t0.0_s0_e-1.jsonl'
            ]
        },
        'gpt-4.1-mini': {
            'fib': ['llm_running/math_evaluation/results_xes/gpt-4.1-mini-seed42/xes/test_cot_-1_seed42_t0.0_s0_e-1.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/gpt-4.1-mini-seed42/xes_opt/test_cot_-1_seed42_t0.0_s0_e-1.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/gpt-4.1-mini-seed42/xes/test_cot_-1_seed42_t0.0_s0_e-1.jsonl',
                'llm_running/math_evaluation/results_xes_opt/gpt-4.1-mini-seed42/xes_opt/test_cot_-1_seed42_t0.0_s0_e-1.jsonl'
            ]
        }
    },
}

reasoning_scores = {
    'xes': {
        'Llama-3.1-8B-Instruct': {
            'fib': ['llm_running/math_evaluation/results_xes/Llama-3.1-8B-Instruct-seed42/xes/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Llama-3.1-8B-Instruct-seed42/xes_opt/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Llama-3.1-8B-Instruct-seed42/xes/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Llama-3.1-8B-Instruct-seed42/xes_opt/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'
            ],
        },
        'Llama-3.1-70B-Instruct': {
            'fib': ['llm_running/math_evaluation/results_xes/Llama-3.1-70B-Instruct-seed42/xes/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Llama-3.1-70B-Instruct-seed42/xes_opt/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Llama-3.1-70B-Instruct-seed42/xes/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Llama-3.1-70B-Instruct-seed42/xes_opt/test_llama3-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'
            ]
        },
        'Mistral-7B-Instruct-v0.3': {
            'fib': ['llm_running/math_evaluation/results_xes/Mistral-7B-Instruct-v0.3-seed42/xes/test_mistral_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Mistral-7B-Instruct-v0.3-seed42/xes_opt/test_mistral_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Mistral-7B-Instruct-v0.3-seed42/xes/test_mistral_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Mistral-7B-Instruct-v0.3-seed42/xes_opt/test_mistral_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'
            ]
        },
        'Qwen2.5-32B-Instruct': {
            'fib': ['llm_running/math_evaluation/results_xes/Qwen2.5-32B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Qwen2.5-32B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Qwen2.5-32B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Qwen2.5-32B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'
            ]
        },
        'Qwen2.5-7B-Instruct': {
            'fib': ['llm_running/math_evaluation/results_xes/Qwen2.5-7B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/Qwen2.5-7B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/Qwen2.5-7B-Instruct-seed42/xes/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl',
                'llm_running/math_evaluation/results_xes_opt/Qwen2.5-7B-Instruct-seed42/xes_opt/test_qwen25-math-cot_-1_seed42_t0.6_s0_e-1_reasoning_eval_{eval_model}.jsonl'
            ]
        },
        'claude-sonnet-4-6': {
            'fib': ['llm_running/math_evaluation/results_xes/claude-sonnet-4-6-seed42/xes/test_cot_-1_seed42_t0.0_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/claude-sonnet-4-6-seed42/xes_opt/test_cot_-1_seed42_t0.0_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/claude-sonnet-4-6-seed42/xes/test_cot_-1_seed42_t0.0_s0_e-1_reasoning_eval_{eval_model}.jsonl',
                'llm_running/math_evaluation/results_xes_opt/claude-sonnet-4-6-seed42/xes_opt/test_cot_-1_seed42_t0.0_s0_e-1_reasoning_eval_{eval_model}.jsonl'
            ]
        },
        'gpt-4.1-mini': {
            'fib': ['llm_running/math_evaluation/results_xes/gpt-4.1-mini-seed42/xes/test_cot_-1_seed42_t0.0_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'mcq': ['llm_running/math_evaluation/results_xes_opt/gpt-4.1-mini-seed42/xes_opt/test_cot_-1_seed42_t0.0_s0_e-1_reasoning_eval_{eval_model}.jsonl'],
            'all': [
                'llm_running/math_evaluation/results_xes/gpt-4.1-mini-seed42/xes/test_cot_-1_seed42_t0.0_s0_e-1_reasoning_eval_{eval_model}.jsonl',
                'llm_running/math_evaluation/results_xes_opt/gpt-4.1-mini-seed42/xes_opt/test_cot_-1_seed42_t0.0_s0_e-1_reasoning_eval_{eval_model}.jsonl'
            ]
        }
    },
}

psr_scores = {
    'xes': {
        'claude-sonnet-4-6': {
            'item_scores': 'output/psr_items/claude-sonnet-4-6.json'
        },
        'Llama-3.1-8B-Instruct': {
            'item_scores': 'output/psr_items/Llama-3.1-8B-Instruct.json'
        },
        'Llama-3.1-70B-Instruct': {
            'item_scores': 'output/psr_items/Llama-3.1-70B-Instruct.json'
        },
        'Mistral-7B-Instruct-v0.3': {
            'item_scores': 'output/psr_items/Mistral-7B-Instruct-v0.3.json'
        },
        'Qwen2.5-32B-Instruct': {
            'item_scores': 'output/psr_items/Qwen2.5-32B-Instruct.json'
        },
        'Qwen2.5-7B-Instruct': {
            'item_scores': 'output/psr_items/Qwen2.5-7B-Instruct.json'
        },
        'gpt-4.1-mini': {
            'item_scores': 'output/psr_items/gpt-4.1-mini.json'

        }
    }
}


