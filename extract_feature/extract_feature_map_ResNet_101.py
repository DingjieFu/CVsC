#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File : extract_feature_map_ResNet_101.py
# @Author : DingjieFu
# @Time : 2024/03/16 08:53:46

import os
import h5py
import pickle
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models.resnet as models
from torch.utils.data import Dataset, DataLoader

import argparse
import warnings
warnings.filterwarnings("ignore")
fileDirPath = "<model path>"
rootPath = "<data root>"
os.environ['TORCH_HOME'] = fileDirPath + '/vit-base'


class MyDataset(Dataset):
    def __init__(self, img_dir, file_paths, dataset_name, transform=None):
        self.matcontent = sio.loadmat(file_paths)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        self.img_dir = img_dir
        self.dataset_name = dataset_name
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx][0]
        if self.dataset_name == "AWA2":
            image_file = os.path.join(self.img_dir, '/'.join(image_file.split('/')[5:]))
        elif self.dataset_name == "CUB":
            image_file = os.path.join(self.img_dir, '/'.join(image_file.split('/')[6:]))
        elif self.dataset_name == "SUN":
            image_file = os.path.join(self.img_dir, '/'.join(image_file.split('/')[7:]))
        else:
            raise ValueError("Unkonwn Dataset!")
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = self.transform(image)
        return image


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset feature extraction")
    """ """
    parser.add_argument('--dataset', default = 'AWA2', help='dataset: AWA2/CUB/SUN')
    parser.add_argument('--inputsize', type=int, default = 224, help='input_imgage_size')
    parser.add_argument('--batch_size', type=int, default = 800, help='Resnet batch size')
    """ """
    parser.add_argument('--data_root', default= rootPath + 'data/dataset', help='Path to data root')
    parser.add_argument('--mat_path', default= rootPath + 'data/dataset/xlsa17/data', help='xlsa17 data')
    parser.add_argument('--w2v_path', default= rootPath + 'data/w2v/', help='w2v path')
    parser.add_argument('--is_save', default= True, help='whether to save hdf5 file')
    args = parser.parse_args()
    return args
args = parse_args()


# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_root = args.data_root +  f"/{args.dataset}"
res101_path = args.mat_path + f"/{args.dataset}/res101.mat"
save_path = args.data_root + f"/{args.dataset}/feature_map_ResNet_101_{args.dataset}_{args.inputsize}.hdf5"
attribute_path = args.w2v_path + f"{args.dataset}_attribute.pkl"
split_path = args.mat_path + f"/{args.dataset}/att_splits.mat"

# ResNet101 model
model_ref = models.resnet101(weights=True)
model_ref.eval()
model_f = nn.Sequential(*list(model_ref.children())[:-2])
model_f.to(device)
model_f.eval()
for param in model_f.parameters():
    param.requires_grad = False

# dataloader
data_transforms = transforms.Compose([
    transforms.Resize(args.inputsize),
    transforms.CenterCrop(args.inputsize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

myDataset = MyDataset(dataset_root, res101_path, args.dataset, data_transforms)
dataset_loader = DataLoader(myDataset,batch_size=args.batch_size, shuffle=False, num_workers=0)

with torch.no_grad():
    all_features = []
    for i_batch, imgs in enumerate(dataset_loader):
        print(i_batch)
        imgs = imgs.to(device)
        features = model_f(imgs)
        all_features.append(features.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)

matcontent = myDataset.matcontent
labels = matcontent['labels'].astype(int).squeeze() - 1

# get sample idx
matcontent = sio.loadmat(split_path)
trainval_loc = matcontent['trainval_loc'].squeeze() - 1
# train_loc = matcontent['train_loc'].squeeze() - 1 #--> train_feature = TRAIN SEEN
# val_unseen_loc = matcontent['val_loc'].squeeze() - 1 #--> test_unseen_feature = TEST UNSEEN
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
att = matcontent['att'].T

original_att = matcontent['original_att'].T


with open(attribute_path, 'rb') as f:
    w2v_att = pickle.load(f)
if args.dataset == "AWA2":
    assert w2v_att.shape == (85,300)
elif args.dataset == "CUB":
    assert w2v_att.shape == (312, 300)
elif args.dataset == "SUN":
    assert w2v_att.shape == (102,300)
else:
    raise ValueError("Unkonwn Dataset!")

print('save w2v_att')

if args.is_save:
    f = h5py.File(save_path, "w")
    f.create_dataset('feature_map', data = all_features, compression = "gzip")
    f.create_dataset('labels', data = labels, compression = "gzip")
    f.create_dataset('trainval_loc', data = trainval_loc, compression = "gzip")
    # f.create_dataset('train_loc', data = train_loc,compression = "gzip")
    # f.create_dataset('val_unseen_loc', data = val_unseen_loc,compression = "gzip")
    f.create_dataset('test_seen_loc', data = test_seen_loc, compression = "gzip")
    f.create_dataset('test_unseen_loc', data = test_unseen_loc, compression = "gzip")
    f.create_dataset('att', data = att, compression = "gzip")
    f.create_dataset('original_att', data = original_att, compression = "gzip")
    f.create_dataset('w2v_att', data = w2v_att, compression = "gzip")
    f.close()
