ROOT_PATH=/app

set -euo pipefail



python llava/eval/eval_batch.py --num-chunks 4  --model-name microsoft/llava-med-v1.5-mistral-7b \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/mimic_cxr_closed_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
    --answers-file ${ROOT_PATH}/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/output/type1/close_ended/mimic_cxr_closed/pred_original.jsonl


python llava/eval/eval_batch.py --num-chunks 4  --model-name microsoft/llava-med-v1.5-mistral-7b \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/rad_vqa_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/VQA_RAD/Image_Folder \
    --answers-file ${ROOT_PATH}/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/output/type1/close_ended/VQA_RAD/pred_original.jsonl 

python llava/eval/eval_batch.py --num-chunks 4  --model-name microsoft/llava-med-v1.5-mistral-7b \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/SLAKE/imgs \
    --answers-file ${ROOT_PATH}/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/output/type1/close_ended/SLAKE/pred_original.jsonl


python llava/eval/eval_batch.py --num-chunks 4  --model-name microsoft/llava-med-v1.5-mistral-7b \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/IU-Chest-X-ray/iu_xray/images \
    --answers-file ${ROOT_PATH}/MedHEval/code/baselines/Med-LVLMs/llava-med-1.5/output/type1/close_ended/IU_Xray/pred_original.jsonl