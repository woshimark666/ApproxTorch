import torch
import approxtorch as at
from ResNet_CIFAR10 import resnet20_cifar10
import torchvision
from MobileNetv2_cifar10 import MobileNetV2

def calib():
    train_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    train_dataset = torchvision.datasets.CIFAR10(root='/dataset', train=True, download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    model = MobileNetV2()
    weights = torch.load('mobilenet_top1_92.830.pth', map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    model = model.cuda()
    save_path = 'mobilenet_top1_92.830_calib.pth'
    at.quant_utils.calibrate_int8(model, train_loader, 0.1, qmethod=('static', 'tensor', 'tensor'), save_path=save_path)



def check_weight():
    path = 'mobilenet_top1_92.830_calib.pth'
    # INSERT_YOUR_CODE
    state_dict = torch.load(path, map_location=torch.device('cuda'))
    for name in state_dict.keys():
        print(name)

def check_freeze():
    model = MobileNetV2()
    lut = at.load_lut('exact.txt')
    qmethod = ('static', 'tensor', 'tensor')
    model = at.convert_model(model, lut, 'int8', qmethod, gradient='ste',
                             gradient_lut=None)
    at.unforze_scale(model)
    print(model)
    

if __name__ == '__main__':
    # calib()
    # check_weight()
    check_freeze()


