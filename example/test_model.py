import torch
import approxtorch as at
import ResNet_CIFAR10 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm



device = torch.device('cuda:0')
model = ResNet_CIFAR10.resnet20_cifar10().to(device)
model.load_state_dict(torch.load('top1_92.94.pth'))
lut = at.load_lut('exact_uint8.txt').to(device)

qtype = 'uint8'
qmethod = ('dynamic', 'tensor', 'tensor')
model = at.convert_model(model, lut, qtype, qmethod, None, 'ste', False).to(device)
print(model)


# Prepare CIFAR-10 test dataset and dataloader
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = datasets.CIFAR10(root='/dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Set model to evaluation mode
model.eval()
criterion = nn.CrossEntropyLoss()


with torch.no_grad():
    total = 0
    top1_correct = 0
    top5_correct = 0
    total_loss = 0.0
    
    for images, labels in tqdm(testloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        
        # Top-1 accuracy
        _, pred = outputs.topk(5, 1, True, True)

        top1_correct += torch.sum(pred[:, :1] == labels.view(-1, 1)).item()
        top5_correct += torch.sum(pred == labels.view(-1, 1)).item()
    
    avg_loss = total_loss / total
    top1_acc = (top1_correct / total) * 100  # Convert to percentage
    top5_acc = (top5_correct / total) * 100  # Convert to percentage

print(f"Test Loss: {avg_loss:.4f}")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")












