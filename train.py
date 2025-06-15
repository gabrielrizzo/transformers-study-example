import model_evaluation
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Prepare your dataset
# Let's create a small example dataset
train_texts = [
    "Estou muito feliz com o resultado!",
    "O serviço foi excelente!",
    "Não gostei nada do atendimento.",
    "O produto é de baixa qualidade.",
    "Foi uma experiência maravilhosa!",
    "Péssimo serviço, não recomendo.",
    "Adorei o filme, foi incrível!",
    "O restaurante estava horrível."
]

train_labels = [1, 1, 0, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Create a dataset
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})

# 2. Load tokenizer and model
model_name = "distilbert-base-multilingual-cased"  # Using multilingual model for Portuguese
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

# 4. Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 5. Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reduced from 8 to 2
    per_device_eval_batch_size=2,   # Reduced from 8 to 2
    gradient_accumulation_steps=4,  # Simulate larger batch size
    num_train_epochs=3,
    weight_decay=0.012,
    save_strategy="epoch",
    load_best_model_at_end=True,
    dataloader_pin_memory=False,    # Disable pin memory to save RAM
    fp16=False,                     # Disable mixed precision (can cause issues on some systems)
    optim="adamw_torch",            # Use torch optimizer (more memory efficient)
    remove_unused_columns=True,     # Remove unused columns to save memory
    logging_steps=1,
    max_grad_norm=1.0,
)

# 6. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_train_dataset,
    train_dataset=tokenized_train_dataset,
    compute_metrics=compute_metrics,
)

# 7. Train the model
trainer.train()

# 8. Save the model
model.save_pretrained("./my_sentiment_model")
tokenizer.save_pretrained("./my_sentiment_model")

# 9. Test the model
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    return predicted_label, probabilities

# Test with some examples
# test_texts = [
#     "O filme foi incrível!",
#     "Não gostei do serviço.",
#     "Foi uma experiência maravilhosa!"
# ]

# for text in test_texts:
#     label, probs = predict_sentiment(text)
#     print(f"\nTexto: {text}")
#     print(f"Sentimento: {'Positivo' if label == 1 else 'Negativo'}")
#     print(f"Probabilidades: {probs}")