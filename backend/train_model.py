
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from bert_embdding import get_bert_embedding
from sklearn.model_selection import train_test_split
label_map = {
    'חומרים ממכרים': 0,
    'בדידות': 1
}

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # raw logits

def train(df, epochs = 10, test_size = 0.35):

    df['label'] = df['תופעות'].map(label_map) 
    X_df = pd.DataFrame(df["text"].apply(lambda x: get_bert_embedding(str(x))).tolist())  # shape: (N, 768)
    # Binary labels
    y_array = df['label'].values
    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_array, test_size=test_size, random_state=42, stratify=y_array
    )
    # convert to pytorch tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    
    model = SimpleClassifier(input_dim=X_df.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = (torch.sigmoid(outputs) > 0.5).int()
        acc = (preds == y_test_tensor.int()).float().mean().item()
        print(f"Test Accuracy: {acc:.2f}")

    return model
