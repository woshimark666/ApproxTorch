import torch
import approxtorch as ap
import torch.nn as nn
import random
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)
        
        # Initialize weights randomly
        self._initialize_weights()
        
    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second Convolutional Block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        
        return x

def test_lenet5():
    # Create a sample input tensor (batch_size=1, channels=1, height=32, width=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 1, 32, 32).to(device)
    # Initialize the model
    model = LeNet5().to(device)
    lut = ap.load_lut('exact.txt').to(device)
    grad_lut = torch.load('accurate_grad.pth').to(device)
    model = ap.convert_model(model, lut, grad_lut, 'ste').to(device)

    # print(model)
    # Forward pass
    output = model(x)
    
    print(output)
    print(output.shape)

if __name__ == "__main__":
    test_lenet5()


