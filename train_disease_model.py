import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os

# 1. Configuration
CSV_PATH = "symbipredict_2022.csv"
MODEL_SAVE_PATH = "app/agent/llms/disease_model.pth"
METADATA_PATH = "app/agent/llms/model_metadata.json"

class DiseaseClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def train():
    print("🚀 Starting training process...")
    
    # 2. Load Data
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found!")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Symptoms are all columns except 'prognosis'
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    symptom_columns = X.columns.tolist()
    
    # 3. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X.values, y_encoded, test_size=0.2, random_state=42)
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # 5. Model Initialization
    input_size = len(symptom_columns)
    model = DiseaseClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Training Loop
    epochs = 50
    print(f"📊 Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = (predicted == y_test_t).sum().item() / y_test_t.size(0)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    # 7. Save Model and Metadata
    print("💾 Saving model and metadata...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    metadata = {
        "symptoms": symptom_columns,
        "diseases": le.classes_.tolist()
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Training Complete!")
    print(f"📍 Model saved to: {MODEL_SAVE_PATH}")
    print(f"📍 Metadata saved to: {METADATA_PATH}")

if __name__ == "__main__":
    train()
