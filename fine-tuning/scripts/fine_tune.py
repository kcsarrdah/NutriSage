import os
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Define paths (adjust paths to your current structure)
MODEL_DIR = "model/"  # Path to the pre-trained model
TRAIN_FILE = "data/nutrisage_train.jsonl"  # Training data
VAL_FILE = "data/nutrisage_val.jsonl"  # Validation data
OUTPUT_DIR = "model/finetuned_model/"  # Directory to save the fine-tuned model

# Load dataset
print("Loading datasets...")
datasets = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

# Load pre-trained model and tokenizer
print("Loading pre-trained model and tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)  # Use the correct tokenizer for the model
model = LlamaForCausalLM.from_pretrained(MODEL_DIR)

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

print("Tokenizing datasets...")
tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=["text"])

# Set training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    warmup_steps=100,
    fp16=True,  # Enable mixed precision for faster training on GPUs
    overwrite_output_dir=True,
    report_to="none",  # Disable reporting to avoid external dependencies
)

# Initialize Trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed!")
