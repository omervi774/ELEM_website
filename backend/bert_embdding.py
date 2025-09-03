from transformers import AutoTokenizer, AutoModel
import torch

# Load Hebrew BERT (only once, globally)
tokenizer = AutoTokenizer.from_pretrained("onlplab/alephbert-base")
model = AutoModel.from_pretrained("onlplab/alephbert-base")

def get_bert_embedding(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get BERT output
    with torch.no_grad():
        outputs = model(**inputs)

    # Use CLS token as sentence embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
    return cls_embedding.squeeze().numpy().tolist()  # convert to list for JSON / CSV