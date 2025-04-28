# training binary classification model

import torch
import torch.nn as nn
import torch.optim as optim
from model import CKDNet
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import random
import numpy as np

# create dataset class
class ChronicKidneyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data['class'] = self.data['class'].apply(lambda x: 1 if x == 'ckd' else 0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data.iloc[idx][['pcv', 'sc', 'hemo']].values.astype(np.float32)
        label = self.data.iloc[idx]['class']
        return torch.tensor(features), torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    batch_size = 32
    learning_rate = 0.001
    num_epochs = 250

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ChronicKidneyDataset('chronic_kidney_data_relevant.csv')
    print(len(dataset))

    # split the dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load the model
    model = CKDNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            outputs = outputs.squeeze()

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average test loss
        test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%')
        
    # save model
    torch.save(model.state_dict(), 'ckd_model.pth')
    print('Training complete. Model saved!')

        






# initalise wandb

# wandb log 

# save all models
