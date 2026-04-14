dataset=mimic_cxr
ROOT_PATH=/app
export CUSTOM_TF=/app/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2
export PYTHONPATH="$CUSTOM_TF/src:${PYTHONPATH}"
export HF_HOME=/app/Models
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava_med_v1.5
# baselines=(original DoLa PAI opera avisc m3id VCD damro)
baselines=(med_opera)
# # conda activate report_eval2
backbone=llava_med_1.5
set -euo pipefail
cd /app/MedHEval/code/baselines/Mitigation/llava-med-1.5


for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 1  --model-name microsoft/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/open-ended/MIMIC-CXR_pairs.json \
        --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
        --answers-file ${ROOT_PATH}/MedHEval/pred/${backbone}/${baseline}/type1/open_ended/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done

for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 1  --model-name microsoft/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json \
        --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
        --answers-file ${ROOT_PATH}/MedHEval/pred/${backbone}/${baseline}/type2/open_ended/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done

