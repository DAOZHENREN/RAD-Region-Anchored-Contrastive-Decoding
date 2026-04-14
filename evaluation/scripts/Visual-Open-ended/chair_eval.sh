#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medheval_report_py37
ROOT_PATH=/app/MedHEval
dataset=mimic_cxr
# first, run all the backbone models

# baselines=(RadFM MiniGPT4 XrayGPT CheXagent LLM-CXR Med-flamingo)
baselines=(med_opera)
# baselines=(DoLa)
backbone=llava_med_1.5

# for baseline in "${baselines[@]}"; do

#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_chair.py \
#         --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#         --pred ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/mimic_open1_res.csv \
#         --eval_res ${ROOT_PATH}/hallucination/${dataset}/open_inference/${baseline}/eval_chair_res.txt \
#     wait
# done

for baseline in "${baselines[@]}"; do

    python /app/MedHEval/code/evaluation/report_eval/run_chair.py \
        --gt /app/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/open-ended/MIMIC-CXR_pairs.csv\
        --pred ${ROOT_PATH}/pred/${backbone}/${baseline}/type1/open_ended/pred_${baseline}.csv \
        --eval_res ${ROOT_PATH}/pred/${backbone}/${baseline}/type1/open_ended/${baseline}_eval_chair_res.txt \
    wait
done

# backbone=(llava_med llava_med_1.5 llava_v1.6 llava_v1.6_13b)

# then run llava-med and llava-med 1.5
# baselines=(original DoLa PAI opera m3id VCD avisc damro) 

# backbone=llava_med
# # backbone=llava_med_1.5

# for baseline in "${baselines[@]}"; do
#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_chair.py \
#         --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#         --pred ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.csv \
#         --eval_res ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_chair_${baseline}.txt
#     wait
# done

# backbone=llava_v1.6
# # backbone=llava_v1.6_13b
# baselines=(original DoLa PAI m3id VCD avisc damro) 
# for baseline in "${baselines[@]}"; do
#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_chair.py \
#         --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#         --pred ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.csv \
#         --eval_res ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_chair_${baseline}.txt
#     wait
# done

# backbone=llava_v1.6
# backbone=llava_v1.6_13b
# baselines=(original DoLa PAI m3id VCD damro) 
# # baselines=(original) 
# for baseline in "${baselines[@]}"; do
#     python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_chair.py \
#         --gt ${ROOT_PATH}/hallucination/${dataset}/type1_open/open_pairs.csv \
#         --pred ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/pred_${baseline}.csv \
#         --eval_res ${ROOT_PATH}/hallucination/MedHEval/type1/baselines_${backbone}/mimic_cxr_open/eval_chair_${baseline}.txt
#     wait
# done