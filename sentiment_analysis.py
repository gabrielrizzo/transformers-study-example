from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("my_sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained("my_sentiment_model")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    return predicted_label, probabilities

text_list = [
    "O filme foi incrível!",
    "Não gostei do serviço.",
    "Foi uma experiência maravilhosa!"
]

# for text in text_list:
#     label, probs = predict_sentiment(text)
#     print(f"\nTexto: {text}")
#     print(f"Sentimento: {'Positivo' if label == 1 else 'Negativo'}")
#     print(f"Probabilidades: {probs}")