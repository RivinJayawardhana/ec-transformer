from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

input_text = "i am ever feeling nostalgic about the fireplace i will know that it is still on the property"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_class = torch.argmax(probabilities).item()
confidence = probabilities[0][predicted_class].item()
labels = ['negative', 'positive']
out = {'class': labels[predicted_class], 'confidence': confidence}
print(out)
