#!/bin/bash
seeds=(4)
ROOT_PATH=/app
export CUSTOM_TF=/app/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2
export PYTHONPATH="$CUSTOM_TF/src:${PYTHONPATH}"
export HF_HOME=/app/Models
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
dataset=SLAKE

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava_med_v1.5
cd /app/MedHEval/code/baselines/Mitigation/llava-med-1.5
set -euo pipefail
# baselines=(original DoLa PAI opera avisc m3id VCD damro)
baselines=(med_opera)
backbone=llava_med_1.5
# # # conda activate report_eval2
# for baseline in "${baselines[@]}"; do
#     python llava/eval/eval_batch.py --num-chunks 2  --model-name microsoft/llava-med-v1.5-mistral-7b \
#         --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json \
#         --image-folder ${ROOT_PATH}/data/sda/SLAKE/imgs \
#         --answers-file ${ROOT_PATH}/MedHEval/pred/${backbone}/${baseline}/type1/close_ended/${dataset}/pred_${baseline}.jsonl \
#         --baseline ${baseline}
#     wait
# done


# dataset=VQA_RAD
# # baselines=(original DoLa PAI opera avisc m3id VCD damro)
# baselines=(med_opera)
# # # conda activate report_eval2
# for baseline in "${baselines[@]}"; do
#     python llava/eval/eval_batch.py --num-chunks 2  --model-name microsoft/llava-med-v1.5-mistral-7b \
#         --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/rad_vqa_pairs.json \
#         --image-folder ${ROOT_PATH}/data/sda/${dataset}/Image_Folder \
#         --answers-file ${ROOT_PATH}/MedHEval/pred/${backbone}/${baseline}/type1/close_ended/${dataset}/pred_${baseline}.jsonl \
#         --baseline ${baseline}
#     wait
# done


# dataset=mimic_cxr
# baselines=(med_opera)
# # # conda activate report_eval2
# for baseline in "${baselines[@]}"; do
#     python llava/eval/eval_batch.py --num-chunks 2  --model-name microsoft/llava-med-v1.5-mistral-7b \
#         --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/mimic_cxr_closed_pairs.json \
#         --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
#         --answers-file ${ROOT_PATH}/MedHEval/pred/${backbone}/${baseline}/type1/close_ended/${dataset}/pred_${baseline}.jsonl \
#         --baseline ${baseline}
#     wait
# done


# dataset=IU_Xray
# baselines=(med_opera)
# # # conda activate report_eval2
# for baseline in "${baselines[@]}"; do
#     python llava/eval/eval_batch.py --num-chunks 2  --model-name microsoft/llava-med-v1.5-mistral-7b \
#         --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json \
#         --image-folder ${ROOT_PATH}/data/sda/IU-Chest-X-ray/iu_xray/images \
#         --answers-file ${ROOT_PATH}/MedHEval/pred/${backbone}/${baseline}/type1/close_ended/${dataset}/pred_${baseline}.jsonl \
#         --baseline ${baseline}
#     wait
# done


dataset=mimic_cxr
baselines=(med_opera)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 2  --model-name microsoft/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json \
        --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
        --answers-file ${ROOT_PATH}/MedHEval/pred/${backbone}/${baseline}/type2/close_ended/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done


dataset=mimic_cxr
baselines=(med_opera)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 2  --model-name microsoft/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json \
        --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
        --answers-file ${ROOT_PATH}/MedHEval/pred/${backbone}/${baseline}/type3/close_ended/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done