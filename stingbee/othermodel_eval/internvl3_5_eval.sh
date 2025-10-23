python stingbee/othermodel_eval/internvl3_5_classfication_eval.py \
  --model-path "/home/data2/zkj/llt_code/public_model/InternVL3_5-1B" \
  --question_file "/home/data2/zkj/llt_code/STING-BEE/stingbee/eval/VQA Evaluation Benchmark/Ours_Pidray_VQA_questions.jsonl" \
  --answers_file "/home/data2/zkj/llt_code/STING-BEE/stingbee/othermodel_eval/results/internvl3_5_xray_answers.jsonl" \
  --batch_size 1 \
  --temperature 0.0 \
  --num_chunks 1 \
  --chunk_idx 0