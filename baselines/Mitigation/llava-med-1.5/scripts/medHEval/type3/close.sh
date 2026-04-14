ROOT_PATH=/app
export CUSTOM_TF=/app/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2
export PYTHONPATH="$CUSTOM_TF/src:${PYTHONPATH}"
export HF_HOME=/app/Models
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava_med_v1.5
cd /app/MedHEval/code/baselines/Mitigation/llava-med-1.5
set -euo pipefail
baselines=(med_opera)
backbone=llava_med_1.5
# baselines=(original DoLa PAI opera avisc m3id VCD damro)
# # conda activate report_eval2
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 2  --model-name microsoft/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Context_Misalignment_Hallucination/MIMIC-CXR_pairs_test.json \
        --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
        --answers-file ${ROOT_PATH}/MedHEval/pred/${backbone}/${baseline}/type3/close_ended/pred_${baseline}.jsonl \
        --baseline ${baseline}
    wait
done

unset CUSTOM_TF
unset PYTHONPATH