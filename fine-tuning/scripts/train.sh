#!/bin/bash
# Paths
MODEL_DIR="fine-tuning/model/"  # Path to your pre-trained model directory
OUTPUT_DIR="fine-tuning/model/finetuned_model/"  # Output directory for fine-tuned model
TRAIN_DATA="fine-tuning/nutrisage_train.jsonl"
VAL_DATA="fine-tuning/nutrisage_val.jsonl"

# Fine-tuning
python3 -m transformers.examples.pytorch.language-modeling.run_clm \
    --model_name_or_path $MODEL_DIR \
    --train_file $TRAIN_DATA \
    --validation_file $VAL_DATA \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --fp16 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --overwrite_output_dir
