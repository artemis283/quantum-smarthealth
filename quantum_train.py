from quantum_circuit import QuantumModel
import torch
import torch.optim as optim
import torch.nn as nn
import random
from train import ChronicKidneyDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    batch_size = 32
    learning_rate = 0.001
    num_epochs = 250

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ChronicKidneyDataset('chronic_kidney_data_relevant.csv')

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = QuantumModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            outputs = torch.sigmoid(outputs)  # since output is [-1, 1], BCELoss needs [0,1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), 'ckd_quantum_model.pth')
    print('Quantum model training complete. Model saved!')




    

        