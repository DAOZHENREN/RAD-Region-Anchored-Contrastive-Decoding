ROOT_PATH=/app

export TRANSFORMERS_CACHE=${ROOT_PATH}/data/sda/model
export HF_HOME=${ROOT_PATH}/data/sda/model
export HF_ENDPOINT=https://hf-mirror.com

python llava/eval/run_med_datasets_eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/data/sda/model/llava-med-model \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json \
    --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
    --question_type close \
    --answers-file ${ROOT_PATH}/MedHEval/benchmark_data/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json