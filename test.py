import sys
package_dir = "./pretrained-models.pytorch-master/"
sys.path.insert(0, package_dir)
import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import pretrainedmodels

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('./dataset/test', self.data.loc[idx, 'id_code'] + '.jpeg')
        image = Image.open(img_name)
        image = self.transform(image)
        return {'image': image}


# model = pretrainedmodels.__dict__['resnet101'](pretrained=None)

# model.avg_pool = nn.AdaptiveAvgPool2d(1)
# model.last_linear = nn.Sequential(
#                           nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                           nn.Dropout(p=0.25),
#                           nn.Linear(in_features=2048, out_features=2048, bias=True),
#                           nn.ReLU(),
#                           nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                           nn.Dropout(p=0.5),
#                           nn.Linear(in_features=2048, out_features=1, bias=True),
#                          )
model = torchvision.models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(2048, 1)
model.load_state_dict(torch.load("./inception_v3_epoch_3_10.pth"))
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False
# print('start model eval')
# model.eval()


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = RetinopathyDatasetTest(csv_file='./dataset/testLabels_cropped.csv',
                                      transform=test_transform)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

# coef = [0.5, 1.5, 2.5, 3.5]
#
# for i, pred in enumerate(test_preds):
#     if pred < coef[0]:
#         test_preds[i] = 0
#     elif pred >= coef[0] and pred < coef[1]:
#         test_preds[i] = 1
#     elif pred >= coef[1] and pred < coef[2]:
#         test_preds[i] = 2
#     elif pred >= coef[2] and pred < coef[3]:
#         test_preds[i] = 3
#     else:
#         test_preds[i] = 4

sample = pd.read_csv("./sample_submission.csv")
sample.diagnosis = test_preds.astype(float)
sample.to_csv("sample_submission.csv", index=False)
