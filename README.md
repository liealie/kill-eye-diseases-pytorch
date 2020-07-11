# kill-eye-diseases-pytorch

## 项目日志
2020.7.10 项目开始在github上进行代码托管
2020.7.11 更新使用方法

## 项目目标：
为了防止世界被破坏，顺便水一些论文，开发基于pytorch的分类算法，对眼底图片进行分级诊断

## 基础代码来源：
数据集来自于kaggle竞赛
https://www.kaggle.com/c/diabetic-retinopathy-detection
https://www.kaggle.com/c/aptos2019-blindness-detection
代码来自于kaggle竞赛开源notebook
https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59

## 使用方法
1.下载文件夹
2.将数据集按dataset/trainLable.csv和dataset/testLable.csv进行分类，存到dataset文件夹中，目录应该是这样的

--root
 |--dataset
   |--train
     |--102left.jpeg
     ...
   |--test
     |--10left.jpeg
     |--...
3.下载预训练模型，在root新建一个pretrained_model文件夹，并将预训练模型放到文件夹内

--root
 |--pretrained_model
   |--resnet50-19c8e357.pth
   |--...
   
4.把train.py里面的路径设置好就可以愉快的训练了
