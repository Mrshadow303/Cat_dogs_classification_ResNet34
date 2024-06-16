import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
# from effnetv2 import effnetv2_s
from torch.autograd import Variable
import os

import plot_curves
 
# 设置超参数
 

DATA_DIR = os.getcwd()                  # 程序文件路径
TRAIN_DIR = DATA_DIR + "/train_set"     # 训练集路径
TEST_DIR = DATA_DIR + "/test_set"       # 测试集路径
BATCH_SIZE = 64                         # 一次训练的图片数量
EPOCHS = 30                             # 训练轮数
modellr = 1e-3                          # 学习率
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       #检查是否有cuda环境，没有则使用cpu
CONTINUE_TRAIN = False                  # 是否使用以前的 pth 文件继续训练？
PTH_FILE = f"{DATA_DIR}/34lr1e-3adam_95.2581%.pth"                      #调用之前生成好的 pth 文件并继续训练
PTH_SAVE_DIR = DATA_DIR                 # pth 文件保存路径      
BEST_ACC = 0.0                          # 最佳正确率
results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

 
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
 
])
transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 读取数据
dataset_train = datasets.ImageFolder(TRAIN_DIR, transform)
# print(dataset_train.imgs)
# 对应文件夹的label
print(f"训练集标签：{dataset_train.class_to_idx}")
dataset_test = datasets.ImageFolder(TEST_DIR, transform_test)
# 对应文件夹的label
print(f"测试集标签：{dataset_test.class_to_idx}")
 
# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

 
# 实例化模型并且移动到GPU
criterion = nn.CrossEntropyLoss()
model = torchvision.models.resnet34(pretrained=True) #加载预训练模型
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(DEVICE)

if CONTINUE_TRAIN:
    print("----> 正在加载导入的模型")
    model.load_state_dict(torch.load(PTH_FILE,map_location=DEVICE))
    print("----> 加载模型完成")

# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(),weight_decay=5e-4, lr=modellr)
# optimizer = optim.SGD(model.parameters(),weight_decay=5e-4, lr=modellr)
 

 
def adjust_learning_rate(optimizer, epoch):
    # 设置学习率每30轮削减10%
    modellrnew = modellr * (0.1 ** (epoch // 30))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew
 
 
# 定义训练过程
 
def train(model, device, train_loader, optimizer, epoch):
    global results
    model.train()
    sum_loss = 0
    correct = 0
    total_num = len(train_loader.dataset)
    print(f"训练集长度：{total_num}", f"训练集批次：{len(train_loader)}")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print_loss = loss.data.item()
        sum_loss += print_loss
        
        # 计算准确度
        _, pred = torch.max(output.data, 1)
        correct += torch.sum(pred == target)

        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
    
    ave_loss = sum_loss / len(train_loader)
    correct = correct.data.item()
    accuracy = correct / total_num
    print('epoch: {}, loss: {:.6f}, accuracy: {:.2f}%'.format(epoch, ave_loss, 100. * accuracy))
    results["train_loss"].append(ave_loss)
    results["train_acc"].append(accuracy)

 
def Test(model, device, test_loader):
    global BEST_ACC
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(f"测试集长度：{total_num}", f"测试集批次：{len(test_loader)}")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        test_acc = correct / total_num
        avg_loss = test_loss / len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, correct, len(test_loader.dataset), 100 * test_acc))
    # 生成 pth 文件
    if test_acc > BEST_ACC:
        BEST_ACC = test_acc
        torch.save(
            model.state_dict(),
            f"{PTH_SAVE_DIR}/34lr1e-3adam_{test_acc*100:.4f}%.pth",     #使用resnet34，学习率为1e-3，优化器为adam
        )
    results["test_loss"].append(avg_loss)
    results["test_acc"].append(test_acc)
 
 
# 训练
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, DEVICE, train_loader, optimizer, epoch)
    Test(model, DEVICE, test_loader)

plot_curves.plot_curves(results)