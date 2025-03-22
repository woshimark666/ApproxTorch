import torch
import approxtorch as ap
from approxtorch.layer import approx_Conv2d_int8, approx_Linear_int8
import torch.nn as nn
import torchvision
import time
class approx_lenet5(nn.Module):
    def __init__(self, lut) -> None:
        super(approx_lenet5, self).__init__()
        
        self.conv1 = approx_Conv2d_int8(in_channels=1, out_channels=6, kernel_size=5, lut=lut)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = approx_Conv2d_int8(in_channels=6, out_channels=16, kernel_size=5, lut=lut)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = approx_Linear_int8(in_features=16*4*4, out_features=120, lut=lut)
        self.fc2 = approx_Linear_int8(in_features=120, out_features=84, lut=lut)
        self.fc3 = approx_Linear_int8(in_features=84, out_features=10, lut=lut)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """
        Randomly initialize all weights in the network.
        """
        for m in self.modules():
            if isinstance(m, approx_Conv2d_int8):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, approx_Linear_int8):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 16*4*4)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x
        
def test_inference():
    device = torch.device("cuda:0")
    lut = ap.load_lut('exact.txt').to(device)
    ap_lenet5 = approx_lenet5(lut).to(device)
    weights = torch.load('lenet5_weights.pth', map_location=device)
    ap_lenet5.load_state_dict(weights, strict=False)
    
    # plase change your own dataset path here
    mnist_test= torchvision.datasets.MNIST(root='/dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())
    
    test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=128, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    ap_lenet5.eval()
   
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss = 0.0
        
        time0 = time.perf_counter()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = ap_lenet5(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        time1 = time.perf_counter()
        # Calculate average loss and accuracy
        test_loss = running_loss / len(test_loader)
        test_accuracy = 100 * correct / total
        
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        print(f'Time cost for inference: {time1 - time0:.2f}s')
        

def test_training():
    device = torch.device('cuda:0')
    lut = ap.load_lut('exact.txt').to(device)
    lenet5 = approx_lenet5(lut).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(lenet5.parameters(), lr=1e-3, momentum=0.9)

    mnist_train = torchvision.datasets.MNIST(root='/dataset', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=128, shuffle=True, num_workers=4)
    
    mnist_test = torchvision.datasets.MNIST(root='/dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=128, shuffle=False, num_workers=4)
    
    for epoch in range(100):
        lenet5.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = lenet5(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Print training loss for this batch
        print(f'Epoch [{epoch+1}/100], Loss: {train_loss / len(train_loader):.4f}')
    
    # Evaluate on test set after each epoch
        lenet5.eval()
        with torch.no_grad():
            test_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = lenet5(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            test_loss = test_loss / len(test_loader)
            test_accuracy = 100 * correct / total
            
            print(f'Epoch [{epoch+1}/100], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
            
            
if __name__ == "__main__":
    test_training()