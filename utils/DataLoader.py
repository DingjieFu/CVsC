#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File : DataLoader.py
# @Author : DingjieFu
# @Time : 2024/03/16 14:02:07
"""
    - load data from dataset
"""
import os
import time
import h5py
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
from sklearn import preprocessing
import torchvision.transforms as transforms
from torch.utils.data import Dataset as DSet
from torch.utils.data import DataLoader as DLoader


class Dataloader():
    def __init__(self, args):
        self.is_balance = args.is_balance
        self.is_pairs = args.is_pairs

        self.device = args.device
        self.inputsize = args.inputsize
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.attr_path = args.attr_path + '/{}/new_des.csv'.format(args.dataset)
        self.res101_path = args.mat_path + f"/{args.dataset}/res101.mat"
        self.img_dir = args.image_root + f"/{args.dataset}/"
        self.getdataset(args)
        self.get_idx_classes()

    def getdataset(self, args):
        """
            - return a dict => self.data
        """
        if args.backbone == "Resnet":
            self.h5path = self.img_dir + f"feature_map_ResNet_101_{args.dataset}_{args.inputsize}.hdf5"
        elif args.backbone == "VIT":
            self.h5path = self.img_dir + f"feature_map_VIT_{args.dataset}_{args.inputsize}.hdf5"
        else:
            raise ValueError("Unkonwn backbone!")
        if os.path.exists(self.h5path):
            pass
        else:
            raise ValueError("Cannot find hdf5 file!")
        self.image_root = args.image_root + "/" + args.dataset + "/"

        tic = time.time()
        hf = h5py.File(self.h5path, 'r')
        feature = np.array(hf.get('feature_map'))  # (11788,2048,7,7) visual
        label = np.array(hf.get('labels'))  # (11788, ) class label
        res101 = sio.loadmat(self.res101_path)
        self.image_files = np.squeeze(res101['image_files'])
        with open(args.w2v_path + f'/{self.dataset}_attribute.pkl', 'rb') as f:
            w2v_att = pickle.load(f)
        self.w2v_att = torch.from_numpy(w2v_att).float().to(args.device)  # attribute semantic vector

        def convert_path(image_files, img_dir):
            new_image_files = []
            for idx in range(len(image_files)):
                image_file = image_files[idx][0]
                if self.dataset == "AWA2":
                    image_file = img_dir + '/'.join(image_file.split('/')[5:])
                elif self.dataset == "CUB":
                    image_file = img_dir + '/'.join(image_file.split('/')[6:])
                elif self.dataset == "SUN":
                    image_file = img_dir + '/'.join(image_file.split('/')[7:])
                else:
                    raise ValueError("Unkonwn Dataset!")
                new_image_files.append(image_file)
            return np.array(new_image_files)
        # change xlsa17 image_path to current path
        self.image_files = convert_path(self.image_files, self.image_root)

        # get idx
        splits_content = sio.loadmat(args.mat_path + f"/{args.dataset}/att_splits.mat")
        self.trainval_loc = splits_content['trainval_loc'].squeeze() - 1
        self.train_loc = splits_content['train_loc'].squeeze() - 1
        self.val_unseen_loc = splits_content['val_loc'].squeeze() - 1
        self.test_seen_loc = splits_content['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = splits_content['test_unseen_loc'].squeeze() - 1
        #  get att
        self.original_att = torch.from_numpy(splits_content['original_att'].T).float().to(args.device)
        self.att = torch.from_numpy(splits_content['att'].T).float().to(args.device)
        train_feature = feature[self.trainval_loc]
        test_seen_feature = feature[self.test_seen_loc]
        test_unseen_feature = feature[self.test_unseen_loc]

        """ min-max norm """
        if args.is_scale:
            scaler = preprocessing.MinMaxScaler()
            train_feature = scaler.fit_transform(train_feature)
            test_seen_feature = scaler.fit_transform(test_seen_feature)
            test_unseen_feature = scaler.fit_transform(test_unseen_feature)

        train_feature = torch.from_numpy(train_feature).float()
        test_seen_feature = torch.from_numpy(test_seen_feature).float()
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float()
        # unseen test sample nums
        self.ntest_unseen = test_unseen_feature.size()[0]

        train_label = torch.from_numpy(label[self.trainval_loc]).long()
        test_unseen_label = torch.from_numpy(label[self.test_unseen_loc]).long()
        test_seen_label = torch.from_numpy(label[self.test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(args.device)
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(args.device)

        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels'] = train_label

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen']['labels'] = test_unseen_label

        self.data['train_seen']['img_path'] = self.image_files[self.trainval_loc]
        self.data['test_seen']['img_path'] = self.image_files[self.test_seen_loc]
        self.data['test_unseen']['img_path'] = self.image_files[self.test_unseen_loc]

        print('Finish loading data in ', time.time() - tic)

    def get_idx_classes(self):
        """
            idx in mat => [[],[],[]]
            - self.idxs_list[i] => idxes for self.seenclasses[i]
        """
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list

    def augment_img_path(self):
        self.matcontent = sio.loadmat(self.res101_path)
        self.image_files = np.squeeze(self.matcontent['image_files'])

        def convert_path(image_files, img_dir):
            new_image_files = []
            for idx in range(len(image_files)):
                image_file = image_files[idx][0]
                image_file = os.path.join(img_dir, '/'.join(image_file.split('/')[5:]))
                new_image_files.append(image_file)
            return np.array(new_image_files)

        self.image_files = convert_path(self.image_files, self.img_dir)

        path = self.img_dir + 'feature_map_ResNet_101_{}_{}.hdf5'.format(self.dataset, self.inputsize)
        hf = h5py.File(path, 'r')

        trainval_loc = np.array(hf.get('trainval_loc'))
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))

        self.data['train_seen']['img_path'] = self.image_files[trainval_loc]
        self.data['test_seen']['img_path'] = self.image_files[test_seen_loc]
        self.data['test_unseen']['img_path'] = self.image_files[test_unseen_loc]

        self.attr_name = pd.read_csv(self.attr_path)['new_des']

    def next_batch_img(self, batch_size, class_id, is_trainset=False):
        features = None
        labels = None
        img_files = None
        if class_id in self.seenclasses:
            if is_trainset:
                features = self.data['train_seen']['resnet_features']
                labels = self.data['train_seen']['labels']
                img_files = self.data['train_seen']['img_path']
            else:
                features = self.data['test_seen']['resnet_features']
                labels = self.data['test_seen']['labels']
                img_files = self.data['test_seen']['img_path']
        elif class_id in self.unseenclasses:
            features = self.data['test_unseen']['resnet_features']
            labels = self.data['test_unseen']['labels']
            img_files = self.data['test_unseen']['img_path']
        else:
            raise Exception("Cannot find this class {}".format(class_id))

        # note that img_files is numpy type !!!!!
        idx_c = torch.squeeze(torch.nonzero(labels == class_id))

        features = features[idx_c]
        labels = labels[idx_c]
        img_files = img_files[idx_c.cpu().numpy()]

        batch_label = labels[:batch_size].to(self.device)
        batch_feature = features[:batch_size].to(self.device)
        batch_files = img_files[:batch_size]
        batch_att = self.att[batch_label].to(self.device)

        return batch_label, batch_feature, batch_files, batch_att

    def next_batch(self):
        if self.is_balance:
            idx = []
            n_samples_class = max(self.batch_size // self.ntrain_class, 1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class), min(self.ntrain_class, self.batch_size),
                                             replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs, n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        elif self.is_pairs:
            idx = []
            n_samples_class = max(self.batch_size // self.ntrain_class, 2)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class), min(self.ntrain_class, self.batch_size//2),
                                             replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs, n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:self.batch_size]

        batch_feature = self.data['train_seen']['resnet_features'][idx].to(self.device)
        batch_label = self.data['train_seen']['labels'][idx].to(self.device)
        batch_att = self.att[batch_label].to(self.device)
        return batch_label, batch_feature, batch_att

    def __len__(self):
        return self.ntrain


class E2EDataloader():
    def __init__(self, args):
        res101 = sio.loadmat(args.mat_path + f"/{args.dataset}/res101.mat")
        self.label = res101['labels'].astype(int).squeeze() - 1
        self.image_files = res101['image_files'].squeeze()
        att_splits = sio.loadmat(args.mat_path + f"/{args.dataset}/att_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        # print("res101.keys:", res101.keys())
        # print("att_splits.keys:", att_splits.keys())
        self.trainval_loc = att_splits['trainval_loc'].squeeze() - 1
        if args.dataset == 'CUB':
            self.train_loc = att_splits['train_loc'].squeeze() - 1
            self.val_unseen_loc = att_splits['val_loc'].squeeze() - 1
        self.original_att = torch.from_numpy(att_splits['original_att'].T).float().to(args.device)
        self.att = torch.from_numpy(att_splits['att'].T).float().to(args.device)
        self.test_seen_loc = att_splits['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = att_splits['test_unseen_loc'].squeeze() - 1
        self.allclasses_name = att_splits['allclasses_names']
        self.attribute = torch.from_numpy(att_splits['att'].T).float()
        self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
        self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        # self.attri_name = att_splits['attri_name']
        with open(args.w2v_path + f"/{args.dataset}_attribute.pkl", 'rb') as f:
            w2v_att = pickle.load(f)
        self.w2v_att = torch.from_numpy(w2v_att).float().to(args.device)


class ImageFilelist(DSet):
    def __init__(self, args, data_inf, transform=None, target_transform=None, image_type=None, select_num=None):
        self.transform = transform
        self.target_transform = target_transform
        if image_type == 'test_unseen_small_loc':
            self.img_loc = data_inf.test_unseen_small_loc
        elif image_type == 'test_unseen_loc':
            self.img_loc = data_inf.test_unseen_loc
        elif image_type == 'test_seen_loc':
            self.img_loc = data_inf.test_seen_loc
        elif image_type == 'trainval_loc':
            self.img_loc = data_inf.trainval_loc
        elif image_type == 'train_loc':
            self.img_loc = data_inf.train_loc
        else:
            raise Exception("choose the image_type in ImageFileList")

        if select_num != None:
            # select_num is the number of images that we want to use
            # shuffle the image loc and choose #select_num images
            np.random.shuffle(self.img_loc)
            self.img_loc = self.img_loc[:select_num]

        self.image_files = data_inf.image_files
        self.image_labels = data_inf.label
        self.imlist = default_flist_reader(args, self.image_files, self.img_loc, self.image_labels)
        self.allclasses_name = data_inf.allclasses_name
        # self.attri_name = data_inf.attri_name

        self.image_labels = self.image_labels[self.img_loc]
        label, idx = np.unique(self.image_labels, return_inverse=True)
        self.image_labels = torch.tensor(idx)

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = Image.open((impath)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, impath

    def __len__(self):
        num = len(self.imlist)
        return num


class CategoriesSampler():
    # migrated from Liu et.al., which works well for CUB dataset
    def __init__(self, label_for_imgs, n_batch=1000, n_cls=16, n_per=3, ep_per_batch=1):
        self.n_batch = n_batch  # batchs for each epoch
        self.n_cls = n_cls  # ways
        self.n_per = n_per  # shots
        self.ep_per_batch = ep_per_batch  # episodes for each batch, defult set 1
        self.cat = list(np.unique(label_for_imgs))
        self.catlocs = {}
        for c in self.cat:
            self.catlocs[c] = np.argwhere(label_for_imgs == c).reshape(-1)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                selected_classes = np.random.choice(
                    self.cat, self.n_cls, replace=False)

                for c in selected_classes:
                    l = np.random.choice(
                        self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)


def get_loader(args, data, Transform=None):
    Transform = transforms.Compose([
        transforms.Resize(args.inputsize),
        transforms.CenterCrop(args.inputsize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_train = ImageFilelist(args, data_inf=data,
                                  transform=Transform,
                                  image_type='trainval_loc')

    if args.train_mode == 'distributed':
        train_label = dataset_train.image_labels
        sampler = CategoriesSampler(
            train_label,
            n_batch=args.n_batch,
            n_cls=args.ways,
            n_per=args.shots
        )
        trainloader = DLoader(dataset=dataset_train, batch_sampler=sampler, num_workers=4, pin_memory=True)
    elif args.train_mode == 'random':
        trainloader = DLoader(
            dataset_train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

    dataset_test_unseen = ImageFilelist(args, data_inf=data,
                                        transform=Transform,
                                        image_type='test_unseen_loc')
    testloader_unseen = DLoader(
        dataset_test_unseen,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    dataset_test_seen = ImageFilelist(args, data_inf=data,
                                      transform=Transform,
                                      image_type='test_seen_loc')
    testloader_seen = DLoader(
        dataset_test_seen,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return trainloader, testloader_unseen, testloader_seen


def default_flist_reader(args, image_files, img_loc, image_labels):
    imlist = []
    image_files = image_files[img_loc]
    image_labels = image_labels[img_loc]
    for image_file, image_label in zip(image_files, image_labels):
        if args.dataset == 'AWA2':
            image_file = args.image_root + '/AWA2/' + '/'.join(image_file[0].split('/')[5:])
        elif args.dataset == 'CUB':
            image_file = args.image_root + '/CUB/' + '/'.join(image_file[0].split('/')[6:])
        elif args.dataset == 'SUN':
            image_file = args.image_root + '/SUN/' + '/'.join(image_file[0].split('/')[7:])
        else:
            raise ValueError("Unkonwn dataset!")
        imlist.append((image_file, int(image_label)))
    return imlist
