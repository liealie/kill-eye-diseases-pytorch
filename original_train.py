import pandas as pd
import argparse
import time
import torchvision
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm

from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
# parser = argparse.ArgumentParser(description='Train a simple classifier for blindness detection')
# parser.add_argument('--train_dir', default='',
#                     help='where you store the train img, e.g./input/aptos2019-blindness-detection/train_images')
# parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
# parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
# parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')
# args = parser.parse_args()
output_features = 1  # 分级数据，0-4级总共5级


# 首先搞一个dataset
class RetinopathyDatasetTrain(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('./dataset/train', self.data.loc[idx, 'image'] + '.jpeg')  # todo: 修正路径
        image = Image.open(img_name)
        image = image.resize((224, 224), resample=Image.BILINEAR)
        label = torch.tensor(self.data.loc[idx, 'level'])
        return {'image': transforms.ToTensor()(image),
                'labels': label
                }


# 设定好model
print('load model structure')
model = torchvision.models.resnet50(pretrained=False)  # todo:如果不想用官方模型，自己写的话需要替换torchvision.models.resnet101()
print('load pretrained model')
model.load_state_dict(torch.load("./pretrained_models/resnet50-19c8e357.pth"))  # todo:修正路径
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, output_features)
print('load model to gpu')
model = model.to(device)

# 将dataset设定好
print('load dataset')
train_dataset = RetinopathyDatasetTrain(csv_file='./dataset/trainLabels_cropped.csv')  # todo: 修正路径
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  # windows情况下需要设成0

plist = [
         {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},
         {'params': model.fc.parameters(), 'lr': 1e-3}
         ]
# 设定优化器
print('set optimizer')
optimizer = optim.Adam(plist, lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10)


# 开始训练
print('start training')
since = time.time()
criterion = nn.MSELoss()
num_epochs = 10
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    scheduler.step()
    model.train()
    running_loss = 0.0
    tk0 = tqdm(data_loader, total=int(len(data_loader)))
    counter = 0
    for bi, d in enumerate(tk0):
        inputs = d["image"]
        labels = d["labels"].view(-1, 1)
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        counter += 1
        tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))
        if counter % 10 == 0:
            print('step{}: '.format(counter) +
                  'training loss: {:.4f}'.format(running_loss / (counter * data_loader.batch_size)))
    epoch_loss = running_loss / len(data_loader)
    print('Training Loss: {:.4f}'.format(epoch_loss))
    time_elapsed_epoch = time.time() - since
    print('Epoch completed in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
torch.save(model.state_dict(), "inception_v3_10.pth")
