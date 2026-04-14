ROOT_PATH=/app

python llava/eval/run_med_datasets_eval_batch.py --num-chunks 1  --model-name ${ROOT_PATH}/data/sda/model/llava-med-model \
    --question-file ${ROOT_PATH}/MedHEval/benchmark_data/Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json \
    --image-folder ${ROOT_PATH}/data/sda/mimic-cxr/files \
    --question_type open \
    --answers-file ${ROOT_PATH}/MedHEval/benchmark_data/Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json