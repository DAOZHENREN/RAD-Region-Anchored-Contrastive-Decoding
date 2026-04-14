#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh

ROOT_PATH=/app/MedHEval
dataset=mimic_cxr
# first, run all the backbone models
export CUSTOM_TF=/app/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2
export HF_HOME=/app/Models
export TRANSFORMERS_CACHE=$HF_HOME/hub
# baselines=(RadFM MiniGPT4 XrayGPT CheXagent LLM-CXR Med-flamingo)
baselines=(med_opera)
backbone=llava_med_1.5

set -euo pipefail

for baseline in "${baselines[@]}"; do
    conda activate medheval_report_py37
    python /app/MedHEval/code/evaluation/report_eval/run_eval.py \
        --gt  /app/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/open-ended/MIMIC-CXR_pairs.csv\
        --pred ${ROOT_PATH}/pred/${backbone}/${baseline}/type1/open_ended/pred_${baseline}.csv \
        --eval_res ${ROOT_PATH}/pred/${backbone}/${baseline}/type1/open_ended/eval_res_${baseline}.csv \
        --report True
    wait
    conda activate medheval_metrics_py310
    python /app/MedHEval/code/evaluation/report_eval/run_all_metrics.py \
            --model_answers_file ${ROOT_PATH}/pred/${backbone}/${baseline}/type1/open_ended/pred_${baseline}.jsonl \
            --eval_res_file ${ROOT_PATH}/pred/${backbone}/${baseline}/type1/open_ended/eval_all_metrics_${baseline}.txt \
            --RadGraphFile ${ROOT_PATH}/pred/${backbone}/${baseline}/type1/open_ended/eval_res_${baseline}.csv

    wait
done

# then run llava-med and llava-med 1.5
# baselines=(original DoLa PAI opera m3id VCD avisc damro) 

# backbone=llava_med
# backbone=llava_med_1.5

# for baseline in "${baselines[@]}"; do
#     # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
#     #     --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#     #     --pred ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.csv \
#     #     --eval_res ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_res_${baseline}.csv \
#     #     --report True
#     # wait
#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
#             --model_answers_file ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.jsonl \
#             --eval_res_file ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_all_metrics_${baseline}.txt \
#             --RadGraphFile ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_res_${baseline}.csv

#     wait
# done

