ROOT_PATH=/app

set -euo pipefail



python llava/eval/run_med_datasets_eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/data/sda/model/llava-med-model \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/mimic_cxr_closed_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
    --answers-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/mimic_cxr_closed_pairs.json 


python llava/eval/run_med_datasets_eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/data/sda/model/llava-med-model \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/rad_vqa_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/VQA_RAD/Image_Folder \
    --answers-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/rad_vqa_pairs.json 

python llava/eval/run_med_datasets_eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/data/sda/model/llava-med-model \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/SLAKE/imgs \
    --answers-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json 


python llava/eval/run_med_datasets_eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/data/sda/model/llava-med-model \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/IU-Chest-X-ray/iu_xray/images \
    --answers-file ${ROOT_PATH}/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json 