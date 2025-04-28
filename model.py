import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

# also try with query tower model architecture

class CKDNet(nn.Module):
    def __init__(self, input_size=3):
        super(CKDNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8) 
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
# testing the model with random input and target
if __name__ == "__main__":
    model = CKDNet(input_size=3)
    print(model)
    criterion = nn.BCELoss()
    inpt = torch.randint(0, 10, (1, 3)).float()
    target = torch.randint(0, 2, (1, 1)).float()
    out = model(inpt)
    loss = criterion(out, target)
    print(f"Loss: {loss.item()}")