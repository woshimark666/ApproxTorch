import torch
import torch.nn as nn
import torch.nn.functional as F
import approxtorch as at

class lenet5(nn.Module):
    def __init__(self) -> None:
        super(lenet5, self).__init__()
  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
            
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 16*4*4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

device = torch.device('cuda:0') 
# load the LUT using ap.load()
lut = at.load_lut('exact.txt')
lut = lut.to(device)

# define the FP32 model
model = lenet5().to(device)
print(model)
# convert FP32 model to approximate model
model = at.convert_model(model, lut, qtype='int8', 
                         qmethod=('dynamic', 'tensor', 'tensor'), 
                         gradient_lut=None, gradient='ste', conv_only=False).to(device)

# check the model again
print(model)

