#!/bin/bash

dataset="xes"
deps_grounding="skill"
q_type="all"
# ['human', 'Llama-3.1-8B-Instruct', 'Llama-3.1-70B-Instruct', 'Mistral-7B-Instruct', 'Qwen2.5-32B-Instruct', 'Qwen2.5-7B-Instruct', 'claude-sonnet-4-6', 'gpt-4.1-mini']
# subject psr

target="subjects"  # model/human-wise psr
for subject in 'human' 'Llama-3.1-8B-Instruct' 'Llama-3.1-70B-Instruct' 'Mistral-7B-Instruct-v0.3' 'Qwen2.5-32B-Instruct' 'Qwen2.5-7B-Instruct' 'claude-sonnet-4-6' 'gpt-4.1-mini' 'Qwen3-Next-80B-A3B-Instruct'
do
  output_path="output/psr_${target}/${subject}_sampled_tmp.json"
  python -m src.evaluate.eval_psr --dataset=${dataset} \
                         --q_type=${q_type} \
                         --deps_grounding=${deps_grounding} \
                         --target=${target} \
                         --subject=${subject} \
                         --output_path=${output_path}
done

#
#target="items"  # item-wise psr
#for subject in 'human' 'Llama-3.1-8B-Instruct' 'Llama-3.1-70B-Instruct' 'Mistral-7B-Instruct-v0.3' 'Qwen2.5-32B-Instruct' 'Qwen2.5-7B-Instruct' 'claude-sonnet-4-6' 'gpt-4.1-mini' 'Qwen3-Next-80B-A3B-Instruct'
#do
#  output_path="output/psr_${target}/${subject}.json"
#  python -m src.evaluate.eval_psr --dataset=${dataset} \
#                         --q_type=${q_type} \
#                         --deps_grounding=${deps_grounding} \
#                         --target=${target} \
#                         --subject=${subject} \
#                         --output_path=${output_path}
#done