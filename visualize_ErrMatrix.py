"""
    visualization of error matrix
"""
import torch
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.DataLoader import Dataloader
from model.causal_model import CVsC_model
from parameter import parse_args
warnings.filterwarnings("ignore")


args = parse_args()
torch.backends.cudnn.benchmark = True
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class ErrorMatrix:
    def __init__(self, groundAtt):
        self.att = groundAtt
        self.errMatrix = np.zeros((self.att.size(0), self.att.size(1)))

    def genErrMatrix(self, predAtt, batchLabel):
        batch_size = predAtt.size(0)
        matrix = np.zeros((self.att.size(0), self.att.size(1)))
        true_atts = self.att[batchLabel].detach().numpy()
        pred_atts = predAtt[np.arange(batch_size), :].detach().numpy()
        matrix[batchLabel] = np.sqrt((pred_atts - true_atts) ** 2)
        return matrix

    def addBatch(self, predAtt, batchLabel):
        assert predAtt.size(0) == batchLabel.size(0)
        self.errMatrix += self.genErrMatrix(predAtt, batchLabel)

    def reset(self):
        self.errMatrix = np.zeros((self.att.size(0), self.att.size(1)))


def l2_norm2(input):
    norm = torch.norm(input, 2, -1, True)
    output = torch.div(input, norm)
    return output


def ErrorMatrix_generate(model, batch_feature, batch_label, errInstance):
    model.eval()
    with torch.no_grad():
        out_package = model(batch_feature)
        errInstance.addBatch(out_package['Pred_att'].cpu(), batch_label.cpu())


def ErrorMatrix_visual(errorMatrix, counts, storName):
    ErrMatrixNorm = errorMatrix / counts
    seenErrMatrix = ErrMatrixNorm[seen_label.cpu()]
    unseenErrMatrix = ErrMatrixNorm[unseen_label.cpu()]
    seenErrMatrix = (seenErrMatrix - torch.min(seenErrMatrix)) / (torch.max(seenErrMatrix)-torch.min(seenErrMatrix))
    unseenErrMatrix = (unseenErrMatrix - torch.min(unseenErrMatrix)) / (torch.max(unseenErrMatrix)-torch.min(unseenErrMatrix))

    seenErrMatrix = np.clip(seenErrMatrix+0.05, 5e-2, 5e-1)
    unseenErrMatrix = np.clip(unseenErrMatrix+0.05, 5e-2, 5e-1)
    ErrMatrixNorm[seen_label.cpu()] = seenErrMatrix
    ErrMatrixNorm[unseen_label.cpu()] = unseenErrMatrix

    new_blues = sns.color_palette('coolwarm', 500)[50:450]
    print(torch.mean(ErrMatrixNorm))
    print("ErrMatrix -> ", ErrMatrixNorm.shape)
    plt.figure(figsize=(18, 16))

    heatmap = sns.heatmap(ErrMatrixNorm, cmap=new_blues,vmin=1e-4-5e-5, vmax=4e-1+5e-2, square=True)
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([0.10, 0.20, 0.30, 0.40])
    cbar.set_ticklabels(['0.10', '0.20', "0.30", "0.40"])
    cbar.ax.tick_params(labelsize=32)
    cbar.ax.set_aspect(20, adjustable='box')
    plt.xlabel('Semantic Index', fontsize=45, labelpad=20)
    plt.ylabel('Class Index', fontsize=45, labelpad=20)
    plt.xticks(ticks=np.arange(0, ErrMatrixNorm.shape[1], 5), labels=np.arange(
        0, ErrMatrixNorm.shape[1], 5), fontsize=24, rotation=0)
    plt.yticks(ticks=np.arange(0, ErrMatrixNorm.shape[0], 5), labels=np.arange(
        0, ErrMatrixNorm.shape[0], 5), fontsize=24)
    plt.gca().tick_params(axis='both', which='major', length=6, width=2)
    plt.tick_params(axis='x', top=True, bottom=False)
    plt.gca().xaxis.set_ticks_position('top')
    sns.despine()
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
    plt.savefig(f'{storName}(all).png', dpi=300)
    plt.close()


"""
    [Pay attention to the file path]
"""
NFS_path = "<Root path>"
dict_list = [NFS_path + f"<weight-name>.pth"]
stor_list = ["<storage-name>"]
dataloader = Dataloader(args)


model = CVsC_model(args, dataloader).to(args.device)
model.load_state_dict(torch.load(dict_list[0]))


if args.dataset == "SUN":
    original_att = dataloader.original_att/1.0
else:
    original_att = dataloader.original_att/100.0
att = l2_norm2(original_att)

err = ErrorMatrix(att.cpu())

test_seen_feature = dataloader.data['test_seen']['resnet_features'].to(args.device)
test_seen_label = dataloader.data['test_seen']['labels'].to(args.device)
test_unseen_feature = dataloader.data['test_unseen']['resnet_features'].to(args.device)
test_unseen_label = dataloader.data['test_unseen']['labels'].to(args.device)
test_feature = torch.cat((test_seen_feature, test_unseen_feature), dim=0)
test_label = torch.cat((test_seen_label, test_unseen_label), dim=0)
seen_label = torch.unique(test_seen_label)
unseen_label = torch.unique(test_unseen_label)
counts = torch.bincount(test_label).reshape(50, 1).cpu()

for i in range(0, test_feature.size(0), args.batch_size):
    batch_label, batch_feature = test_label[i:i+args.batch_size], test_feature[i:i+args.batch_size]
    ErrorMatrix_generate(model, batch_feature, batch_label, err)

ErrorMatrix_visual(err.errMatrix, counts, stor_list[0])
