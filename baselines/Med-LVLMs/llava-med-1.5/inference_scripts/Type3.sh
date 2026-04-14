
ROOT_PATH=/app

# python llava/eval/eval_batch.py --num-chunks 1  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
#     --question-file ${ROOT_PATH}/path/to/Type3/file \
#     --image-folder ${ROOT_PATH}/path/to/images \
#     --answers-file ${ROOT_PATH}/path/to/answer


python llava/eval/eval_batch.py --num-chunks 4  --model-name microsoft/llava-med-v1.5-mistral-7b \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
    --answers-file /app/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/output/type3/pred_original.jsonl