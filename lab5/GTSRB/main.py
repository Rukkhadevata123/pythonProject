from __future__ import print_function
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# 参数设置
class args:
    data = "./data"
    batch_size = 64
    epochs = 40
    lr = 0.0001
    seed = 1
    log_interval = 10

torch.manual_seed(args.seed)

use_gpu = torch.cuda.is_available()
print("Using GPU" if use_gpu else "Using CPU")

# 数据初始化
def initialize_data(folder):
    train_folder = folder + '/train_images'
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            os.mkdir(val_folder + '/' + dirs)
            for f in os.listdir(train_folder + '/' + dirs):
                if f[6:11] in ('00000', '00001', '00002'):
                    os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)

def char_order2int_order(charoder):
    a = [str(i) for i in range(43)]
    a.sort()
    kv = {i: int(a[i]) for i in range(43)}
    return kv[charoder]

def processing_data(data_path):
    """
    数据处理
    :param data_path: 数据集路径
    :return: train_loader, val_loader: 处理后的训练集数据、验证集数据
    """
    initialize_data(data_path)

    # 数据增强和预处理
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.RandomRotation(10),  # 随机旋转
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机颜色抖动
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    # 验证集只进行基本的预处理
    val_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_path + '/train_images', transform=data_transforms, target_transform=char_order2int_order),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_path + '/val_images', transform=val_transforms, target_transform=char_order2int_order),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu
    )

    return train_loader, val_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_model(train_loader, val_loader, save_model_path):
    model = Net()
    if use_gpu:
        model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    res = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = 0
        training_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target, reduction="sum")
            loss.backward()
            optimizer.step()
            max_index = output.max(dim=1)[1]
            correct += (max_index == target).sum().item()
            training_loss += loss.item()
        print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
        res["loss"].append(training_loss / len(train_loader.dataset))
        res["accuracy"].append(100. * correct / len(train_loader.dataset))

        model.eval()
        validation_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                if use_gpu:
                    data = data.cuda()
                    target = target.cuda()
                output = model(data)
                validation_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss / len(val_loader.dataset), correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        res["val_loss"].append(validation_loss / len(val_loader.dataset))
        res["val_accuracy"].append(100. * correct / len(val_loader.dataset))

        model_file = save_model_path + 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file, _use_new_zipfile_serialization=False)
        print('\nSaved model to ' + model_file)

    print("模型训练总时长：", time.time() - start)

    with open(save_model_path + "res", "wb") as f:
        pickle.dump(res, f)

    return model

def evaluate_model(test_loader, save_model_path):
    model = Net()
    model.load_state_dict(torch.load(save_model_path + f'model_{args.epochs}.pth'))
    if use_gpu:
        model.cuda()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))


# model_path = f'results/model_{args.epochs}.pth'  # 请根据实际情况修改路径
# model = Net()
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()

# if use_gpu:
#     model.cuda()

# # ---------------------------------------------------------------------------

# def predict(img):
#     """
#     加载模型和模型预测
#     主要步骤:
#         1.图片处理
#         2.用加载的模型预测图片的类别
#     :param img: PIL.Image 对象
#     :return: int, 模型识别图片的类别
#     """
#     # -------------------------- 实现模型预测部分的代码 ---------------------------
#     # 获取图片的类别
#     # 把图片转换成为tensor
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
#     ])
#     img = transform(img).unsqueeze(0)  # 增加一个维度以匹配批处理的输入格式

#     if use_gpu:
#         img = img.cuda()

#     # 获取输入图片的类别
#     with torch.no_grad():
#         output = model(img)
#         y_predict = output.max(1, keepdim=True)[1].item()

#     # -------------------------------------------------------------------------

#     # 返回图片的类别
#     return y_predict

def main():
    data_path = "./data"  # 数据集路径
    save_model_path = "./results/"  # 保存模型路径和名称

    # 获取数据
    train_loader, val_loader = processing_data(data_path)

    # 创建、训练和保存模型
    model = train_model(train_loader, val_loader, save_model_path)

    # 评估模型
    evaluate_model(val_loader, save_model_path)

if __name__ == '__main__':
    from PIL import Image
    main()
    img_path = '12603.png'  # 替换为你的图片路径
    # img = Image.open(img_path)
    # predicted_class = predict(img)
    # print(f'Predicted class: {predicted_class}')