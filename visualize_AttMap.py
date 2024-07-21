"""
    visualize attention map
"""
import os
import torch
import warnings
import numpy as np
from PIL import Image
import skimage.transform
import matplotlib.pyplot as plt
from torchvision import transforms
from model.causal_model import CVsC_model
from utils.DataLoader import Dataloader
from parameter import parse_args
warnings.filterwarnings("ignore")


data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


def dazle_visualize_attention_np_global_224_small(img_ids, alphas_1, alphas_2, attr_name, idxs_top_p, save_path=None):
    #  alphas_1: [bir]     alphas_2: [bi]
    n = img_ids.shape[0]
    image_size = 224  # one side of the img
    assert alphas_1.shape[1] == alphas_2.shape[1] == len(attr_name)
    r = alphas_1.shape[2]
    h = w = int(np.sqrt(r))
    for i in range(n):
        fig = plt.figure(i, figsize=(30, 15))
        file_path = img_ids[i]
        img_name = file_path.split("/")[-1]
        alpha_1 = alphas_1[i]  # [ir]
        alpha_2 = alphas_2[i]  # [i]

        # Plot original image
        image = Image.open(file_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = data_transforms(image)
        image = image.permute(1, 2, 0)  # [224,244,3] <== [3,224,224]

        idx = 1
        ax = plt.subplot(3, 10, 1)
        idx += 1
        plt.imshow(image)
        plt.axis('off')

        for _, idx_attr in enumerate(idxs_top_p):
            ax = plt.subplot(3, 10, idx)
            idx += 1
            plt.imshow(image)
            alp_curr = alpha_1[idx_attr, :].reshape(7, 7)
            alp_img = skimage.transform.pyramid_expand(
                alp_curr, upscale=image_size/h, sigma=10)
            plt.imshow(alp_img, alpha=0.5, cmap='jet')
            ax.set_title("{}\n(Score = {:.2f})".format(attr_name[idx_attr.item()].title().replace(
                ' ', ''), alpha_2[idx_attr.item()]), {'fontsize': 18})
            plt.axis('off')

        fig.tight_layout()
        os.makedirs(NFS_path + "visualization/", exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path+img_name, dpi=200)
        plt.close()


def AttentionMap_visual(model, batch_feature, batch_files, attrName, idxs_top, storName):
    model.eval()
    with torch.no_grad():
        out_package = model(batch_feature)
    # attention map
    dazle_visualize_attention_np_global_224_small(batch_files,out_package['A'].cpu().numpy(),
                                                    out_package['Pred_att'].cpu().numpy(),
                                                    attrName,
                                                    idxs_top, 
                                                    NFS_path + f'visualization/{storName}/')


args = parse_args()
dataloader = Dataloader(args)

"""
    [Pay attention to the file path]
"""
NFS_path = "<Root path>"
dict_list = [NFS_path + f"<weight-name>.pth"]
stor_list = ["<storage-name>"]


model = CVsC_model(args, dataloader).to(args.device)
model.load_state_dict(torch.load(dict_list[0]))


dataloader.augment_img_path()
file_list = [
    'Laysan_Albatross_0037_699',
    'Sooty_Albatross_0010_796355',
    'Crested_Auklet_0077_785257',
    'Red_Winged_Blackbird_0014_3761',
    'Rusty_Blackbird_0076_6716',
    'Bobolink_0059_10041',
    'Lazuli_Bunting_0093_15030',
    'Painted_Bunting_0086_16540',
    'Gray_Catbird_0032_21551',
    'Red_Faced_Cormorant_0056_796297',
    'Pelagic_Cormorant_0102_23778',
    'Fish_Crow_0065_25942',
    'Scissor_Tailed_Flycatcher_0008_41670',
    'Nighthawk_0041_82183',
    'Brown_Pelican_0036_93843',
    'Tropical_Kingbird_0003_69852',
    'Mallard_0004_76958',
    'Orchard_Oriole_0034_91825',
    'American_Pipit_0035_100181',
    'Savannah_Sparrow_0028_119982'
]


atts = dataloader.att

for i, filename in enumerate(file_list):
    for i, id in enumerate(dataloader.seenclasses):
        id = id.item()
        (batch_label, batch_feature, batch_files, batch_att) = dataloader.next_batch_img(
            batch_size=10, class_id=id, is_trainset=False)
        if filename not in str(batch_files):
            continue
        ground_att = atts[id, :]
        idxs_top = np.argsort(-ground_att.cpu())[:29]
        idx = [filename in str(f) for f in batch_files]
        batch_feature = batch_feature[idx]
        batch_files = batch_files[idx]
        AttentionMap_visual(model, batch_feature, batch_files,  dataloader.attr_name, idxs_top, stor_list[0])


    for i, id in enumerate(dataloader.unseenclasses):
        id = id.item()
        (batch_label, batch_feature, batch_files, batch_att) = dataloader.next_batch_img(
            batch_size=20, class_id=id, is_trainset=False)
        if filename not in str(batch_files):
            continue
        ground_att = atts[id, :]
        idxs_top = np.argsort(-ground_att.cpu())[:29]
        idx = [filename in str(f) for f in batch_files]
        batch_feature = batch_feature[idx]
        batch_files = batch_files[idx]
        AttentionMap_visual(model, batch_feature, batch_files,  dataloader.attr_name, idxs_top, stor_list[0])
