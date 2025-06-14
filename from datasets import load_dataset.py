from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
ds = load_dataset("noob123/imdb_review_3000")

# Convert sentiment to int label early if it's not already
def encode_labels(example):
    example["label"] = 1 if example["sentiment"] == "positive" else 0
    return example

ds = ds.map(encode_labels)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["review"], padding="max_length", truncation=True)

tokenized_ds = ds.map(tokenize_function, batched=True)

# Split dataset
tokenized_ds = tokenized_ds["train"].train_test_split(test_size=0.1)

# Set torch format after all preprocessing
tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Training arguments (optimized)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,  # Increase if GPU memory allows
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),  # Enable mixed precision if using GPU
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
)

# Train
trainer.train()