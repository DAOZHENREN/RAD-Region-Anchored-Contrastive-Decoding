ROOT_PATH=/app

python llava/eval/eval_batch.py --num-chunks 4  --model-name microsoft/llava-med-v1.5-mistral-7b \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
    --question_type open \
    --answers-file ${ROOT_PATH}/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/output/type2/close_ended/pred_original.jsonl