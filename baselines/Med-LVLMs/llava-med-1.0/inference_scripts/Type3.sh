
ROOT_PATH=/app

export TRANSFORMERS_CACHE=${ROOT_PATH}/data/sda/model
export HF_HOME=${ROOT_PATH}/data/sda/model
export HF_ENDPOINT=https://hf-mirror.com

python llava/eval/run_med_datasets_eval_batch.py --num-chunks 1  --model-name ${ROOT_PATH}/data/sda/model/llava-med-model \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
    --answers-file ${ROOT_PATH}/MedHEval/benchmark_data/Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json