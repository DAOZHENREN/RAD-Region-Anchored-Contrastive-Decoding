#!/bin/bash
ROOT_PATH=/app/MedHEval
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medheval_metrics_py310

export CUSTOM_TF=/app/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2
export HF_HOME=/app/Models
export TRANSFORMERS_CACHE=$HF_HOME/hub
dataset=mimic_cxr
# baselines=(original DoLa PAI opera avisc m3id VCD damro)
baselines=(med_opera)
backbone=llava_med_1.5

for baseline in "${baselines[@]}"; do
    python /app/MedHEval/code/evaluation/open_ended_evaluation/generation_metrics.py \
        --model_answers_file ${ROOT_PATH}/pred/${backbone}/${baseline}/type2/open_ended/pred_${baseline}.jsonl \
        --eval_res_file ${ROOT_PATH}/pred/${backbone}/${baseline}/type2/open_ended/eval_all_metrics_${baseline}.txt
    wait
done