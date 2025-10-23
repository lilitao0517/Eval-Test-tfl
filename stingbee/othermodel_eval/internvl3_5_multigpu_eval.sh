
NUM_GPUS=6
QUESTION_FILE="/home/data2/zkj/llt_code/STING-BEE/stingbee/othermodel_eval/classfication_benchmark/opixray_cls_questions.jsonl"
ANSWER_DIR="/home/data2/zkj/llt_code/STING-BEE/stingbee/othermodel_eval/results/classfication"
MODEL_PATH="/home/data2/zkj/llt_code/public_model/InternVL3_5-8B"
for ((i=0; i<$NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i \
  python stingbee/othermodel_eval/internvl3_5_classfication_eval.py \
    --model-path $MODEL_PATH \
    --question_file $QUESTION_FILE \
    --answers_file "$ANSWER_DIR/internvl3_5-8B_opi_part_$i.jsonl" \
    --num_chunks $NUM_GPUS \
    --chunk_idx $i \
    --batch_size 1 \
    --temperature 0.1 \
    > "$ANSWER_DIR/log_$i.txt" 2>&1 &
done

wait
echo "All processes finished. Merging results..."
cat "$ANSWER_DIR"/internvl3_5-8B_opi_part_*.jsonl | sort -k1 > "$ANSWER_DIR/internvl3_5-8B_opi.jsonl"
echo "Final result save finish"