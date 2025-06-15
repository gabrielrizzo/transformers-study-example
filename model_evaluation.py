from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from sentiment_analysis import predict_sentiment

# 1. Load your trained model and tokenizer
model_name = "./my_sentiment_model"  # or your model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. Create a prediction function that SHAP can use
def f(x):
    # Convert numpy array to string if needed
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, str):
        x = [x]
    # Tokenize the input texts
    tv = tokenizer(x, padding=True, truncation=True, return_tensors="pt", max_length=128)
    # Move to the same device as the model
    tv = {k: v.to(model.device) for k, v in tv.items()}
    # Get predictions
    outputs = model(**tv)
    scores = outputs.logits
    # Convert to numpy array with detach()
    return scores.detach().cpu().numpy()

# 3. Create a SHAP explainer with a proper masker
# First, create a background dataset
background_texts = [
    "O serviço foi bom.",
    "Não gostei do atendimento.",
    "Foi uma experiência normal.",
    "O produto é regular."
]

# Create the explainer with a proper masker
explainer = shap.Explainer(
    f,
    tokenizer,
    output_names=["Negativo/Neutro", "Positivo"],
    max_evals=100  # Limit the number of evaluations for faster computation
)

# 4. Generate explanations for some texts
texts_to_explain = [
    "O filme foi incrível e o atendimento foi excelente!",
    #"Não gostei nada do serviço, foi muito ruim."
]

# Get SHAP values for all texts at once
shap_values = explainer(texts_to_explain)

# Get predicted labels for each text
predicted_labels = []
for text in texts_to_explain:
    pred, _ = predict_sentiment(text)  # Unpack the tuple
    predicted_labels.append(pred)

# 5. Visualize the results
shap.plots.text(shap_values)

# Create a DataFrame with SHAP values and tokens
tokens_list = []
shap_values_list = []
text_labels = []  # To store which text each token came from
for i, text in enumerate(texts_to_explain):
    tokens = tokenizer.tokenize(text)
    values = shap_values[i].values
    for token, value in zip(tokens, values):
        tokens_list.append(token)
        # Convert numpy value to Python float
        shap_values_list.append(float(value[0]))
        text_labels.append(f"Text {i+1} ({predicted_labels[i]})")

# Create DataFrame
df = pd.DataFrame({
    'Token': tokens_list,
    'SHAP Value': shap_values_list,
    'Text': text_labels
})

# Create the plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 8))
sns.barplot(data=df, y='Token', x='SHAP Value', hue='Text', orient='h')
plt.xticks(rotation=45, ha='right')
plt.title('Token Importance from SHAP Values')
plt.tight_layout()
sns.despine(left=True, bottom=True)

# Save the plot
plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Print the DataFrame
print("\nSHAP Values DataFrame:")
print(df)

# 6. If you want to see the importance of each token
for i, text in enumerate(texts_to_explain):
    print(f"\nExplanation for: {text}")
    # Get the tokens and their SHAP values
    tokens = tokenizer.tokenize(text)
    values = shap_values[i].values
    
    # Print each token and its contribution
    for token, value in zip(tokens, values):
        print(f"{token}: {value}")