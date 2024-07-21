import os
import time
import tqdm
import torch
import logging
import warnings
import numpy as np
import torch.optim as optim
import torchvision.models.resnet as models
from transformers import ViTFeatureExtractor, ViTModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from parameter import parse_args
from model.causal_model import CVsC_model
from utils.DataLoader import Dataloader, E2EDataloader, get_loader
from utils.helper_func import eval_zs_gzsl, eval_train_acc, eval_zs_gzsl_e2e, eval_train_acc_e2e
warnings.filterwarnings("ignore")


def create_unique_folder_name(base_folder_path):
    count = 0
    new_folder_name = base_folder_path
    while os.path.exists(new_folder_name):
        count += 1
        new_folder_name = f"{base_folder_path}({count})"
    return new_folder_name


def train(args):
    torch.backends.cudnn.benchmark = True
    # ---------- set random seed ----------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # ---------- run log ----------
    os.makedirs(args.log_root_path, exist_ok=True)
    outlogDir = "{}/{}".format(args.log_root_path, args.dataset)
    os.makedirs(outlogDir, exist_ok=True)
    num_exps = len([f.path for f in os.scandir(outlogDir) if f.is_dir()])
    outlogPath = os.path.join(outlogDir, create_unique_folder_name(outlogDir + f"/exp{num_exps}"))
    os.makedirs(outlogPath, exist_ok=True)
    t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    args.log = outlogPath + "/" + t + '.txt'
    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=args.log,
                        filemode='w')
    logger = logging.getLogger(__name__)
    argsDict = args.__dict__
    for eachArg, value in argsDict.items():
        logger.info(eachArg + ':' + str(value))
    logger.info("="*50)

    # ---------- dataset & model ----------
    dataloader = Dataloader(args)
    model = CVsC_model(args, dataloader).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ======================================== train pipeline ======================================== #
    # ----------  epoch  ----------
    batch_num = dataloader.ntrain // args.batch_size
    best_performance = [0.0, 0.0, 0.0, 0.0]
    loss_record = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for epoch in range(1, args.nepoch + 1):
        print('=' * 50)
        print('Epoch: {}'.format(epoch))
        torch.cuda.empty_cache()
        # ----------  train  ----------
        progress = tqdm.tqdm(total=batch_num * args.batch_size,ncols=100, desc='Train {}'.format(epoch))

        for _ in range(batch_num):
            progress.update(args.batch_size)
            model.train()
            optimizer.zero_grad()
            batch_label, batch_feature, _ = dataloader.next_batch()
            batch_output = model(batch_feature)
            batch_output['batch_label'] = batch_label
            batch_loss = model.compute_loss(batch_output)
            loss = batch_loss['loss']
            loss_record[0] += loss.item()
            loss_record[1] += batch_loss['loss_CE'].item()
            loss_record[2] += batch_loss['loss_Cali'].item()
            loss_record[3] += batch_loss['loss_CCL'].item()
            loss_record[4] += batch_loss['loss_SCI'].item()
            loss_record[5] += batch_loss['loss_TCI'].item()
            loss.backward()
            optimizer.step()
        progress.close()
        print(f'Train Loss => loss: {loss_record[0]/batch_num:.5f}, loss_CE: {loss_record[1]/batch_num:.5f}, loss_Cali: {loss_record[2]/batch_num:.5f}, loss_CCL: {loss_record[3]/batch_num:.5f}, loss_SCI: {loss_record[4]/batch_num:.5f}, loss_TCI: {loss_record[5]/batch_num:.5f}')
        # ----------  val&test  ----------
        acc_train = eval_train_acc(dataloader, model, args.device)
        S, U, H, CZSL = eval_zs_gzsl(dataloader, model, args.device)

        if CZSL > best_performance[3]:
            best_performance[3] = CZSL
        if H > best_performance[2]:
            best_performance[:3] = [S, U, H]
            if args.save:
                model_save_path = f"{outlogPath}/best_weight.pth"
                torch.save(model.state_dict(), model_save_path)
                print('model saved to:', model_save_path)

        dict1 = {'epoch': epoch, 'S': S, 'U': U, 'H': H,'CZSL': CZSL, 'acc_train': acc_train}    
        dict2 = {'S': best_performance[0], 'U': best_performance[1],'H': best_performance[2],
                 'CZSL': best_performance[3]}
        print(f'Performance => S: {dict1["S"]:.5f}, U: {dict1["U"]:.5f}, H: {dict1["H"]:.5f}, CZSL: {dict1["CZSL"]:.5f}, Acc_train: {dict1["acc_train"]:.5f}')
        print(f'Best GZSL|CZSL => S: {dict2["S"]:.5f}, U: {dict2["U"]:.5f}, H: {dict2["H"]:.5f}, CZSL: {dict2["CZSL"]:.5f}')
        logger.info(f'Epoch: {dict1["epoch"]}')
        logger.info("Performance => S:{:.5f}; U:{:.5f}; H:{:.5f}; CZSL:{:.5f}; Acc_train:{:.5f}".format(
            dict1['S'], dict1['U'], dict1['H'], dict1['CZSL'], dict1['acc_train']))
        logger.info('Best GZSL|CZSL => S:{:.5f}; U:{:.5f}; H:{:.5f}; CZSL:{:.5f}'.format(
            best_performance[0], best_performance[1], best_performance[2], best_performance[3]))
        logger.info("-"*50)
        loss_record = [0, 0, 0, 0, 0, 0]
    print("---------- End ----------")


def end2end_train(args):
    torch.backends.cudnn.benchmark = True
    # ---------- set random seed ----------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # ---------- run log ----------
    os.makedirs(args.log_root_path, exist_ok=True)
    outlogDir = "{}/{}".format(args.log_root_path, args.dataset)
    os.makedirs(outlogDir, exist_ok=True)
    num_exps = len([f.path for f in os.scandir(outlogDir) if f.is_dir()])
    outlogPath = os.path.join(outlogDir, create_unique_folder_name(outlogDir + f"/exp{num_exps}"))
    os.makedirs(outlogPath, exist_ok=True)
    t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    args.log = outlogPath + "/" + t + '.txt'
    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=args.log,
                        filemode='w')
    logger = logging.getLogger(__name__)
    argsDict = args.__dict__
    for eachArg, value in argsDict.items():
        logger.info(eachArg + ':' + str(value))
    logger.info("="*50)

    # ---------- dataset & model ----------
    if args.backbone.lower() == "resnet":
        model_ref = models.resnet101(weights=True)
        backbone = torch.nn.Sequential(*list(model_ref.children())[:-2]).to(args.device)
    elif args.backbone.lower() == "vit":
        # ViT_featureExtractor = "/data/fudingjie/model/vit-base/vit-base-patch16-224-in21k"
        ViT_model = "/data/fudingjie/model/vit-base/vit-base-patch16-224"
        # processor = ViTFeatureExtractor.from_pretrained(ViT_featureExtractor)
        model_ref = ViTModel.from_pretrained(ViT_model)
        backbone = torch.nn.Sequential(*list(model_ref.children())[:-2]).to(args.device)
    else:
        raise ValueError("Unknown backbone!")
    
    dataloader = E2EDataloader(args)
    trainloader, testloader_unseen, testloader_seen = get_loader(args, dataloader)
    optimizer_backbone = optim.Adam(backbone.parameters(), lr=1e-4, betas=(0.5, 0.999))
    scheduler_backbone = CosineAnnealingLR(optimizer_backbone, T_max=10, eta_min=1e-5)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    model = CVsC_model(args, dataloader).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # ======================================== train pipeline ======================================== #
    # ----------  epoch  ----------
    batch_num = len(trainloader)
    best_performance = [0.0, 0.0, 0.0, 0.0]
    loss_record = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for epoch in range(1, args.nepoch + 1):
        print('=' * 50)
        print('Epoch: {}'.format(epoch))
        torch.cuda.empty_cache()
        backbone.train()
        model.train()
        if epoch < args.nepoch // 2:
            for param in backbone.parameters():
                param.requires_grad = False
        elif epoch == args.nepoch // 2:
            for param in backbone.parameters():
                param.requires_grad = True
        if epoch >= args.nepoch // 2:
            scheduler_backbone.step()
        # ----------  train  ----------
        progress = tqdm.tqdm(total=batch_num, ncols=100, desc='Train {}'.format(epoch))
        for _, (batch_input, batch_target, _) in enumerate(trainloader):
            progress.update(1)
            optimizer_backbone.zero_grad()
            optimizer.zero_grad()
            batch_input = batch_input.to(args.device)
            if args.backbone.lower() == "vit":
                batch_input = backbone(batch_input).last_hidden_state
            elif args.backbone.lower() == "resnet":
                batch_input = backbone(batch_input)
            else:
                raise ValueError("Unknown backbone!")
            batch_target = batch_target.to(args.device)
            batch_output = model(batch_input)
            batch_output['batch_label'] = batch_target

            batch_loss = model.compute_loss(batch_output)
            loss = batch_loss['loss']
            loss_record[0] += loss.item()
            loss_record[1] += batch_loss['loss_CE'].item()
            loss_record[2] += batch_loss['loss_Cali'].item()
            loss_record[3] += batch_loss['loss_CCL'].item()
            loss_record[4] += batch_loss['loss_SCI'].item()
            loss_record[5] += batch_loss['loss_TCI'].item()
            loss.backward()
            optimizer_backbone.step()
            optimizer.step()
        progress.close()
        print(f'Train Loss => loss: {loss_record[0]/batch_num:.5f}, loss_CE: {loss_record[1]/batch_num:.5f}, loss_Cali: {loss_record[2]/batch_num:.5f}, loss_CCL: {loss_record[3]/batch_num:.5f}, loss_SCI: {loss_record[4]/batch_num:.5f}, loss_TCI: {loss_record[5]/batch_num:.5f}')
        
        # ----------  val&test  ----------
        acc_train = eval_train_acc_e2e(trainloader, backbone, model, dataloader.seenclasses, args.device, args.backbone)
        S, U, H, CZSL = eval_zs_gzsl_e2e(testloader_seen, testloader_unseen, backbone, model, dataloader.seenclasses, dataloader.unseenclasses, args.device, args.backbone)
        if CZSL > best_performance[3]:
            best_performance[3] = CZSL
        if H > best_performance[2]:
            best_performance[:3] = [S, U, H]
            if args.save:
                backbone_save_path = f"{outlogPath}/backbone.pth"
                torch.save(backbone.state_dict(), backbone_save_path)
                print('backbone saved to:', backbone_save_path)
                classifier_save_path = f"{outlogPath}/classifier.pth"
                torch.save(model.state_dict(), classifier_save_path)
                print('classifier saved to:', classifier_save_path)
        
        dict1 = {'epoch': epoch, 'S': S, 'U': U, 'H': H,'CZSL': CZSL, 'acc_train': acc_train}    
        dict2 = {'S': best_performance[0], 'U': best_performance[1],'H': best_performance[2], 
                 'CZSL': best_performance[3]}
        print(f'Performance => S: {dict1["S"]:.5f}, U: {dict1["U"]:.5f}, H: {dict1["H"]:.5f}, CZSL: {dict1["CZSL"]:.5f}, Acc_train: {dict1["acc_train"]:.5f}')
        print(f'Best GZSL|CZSL => S: {dict2["S"]:.5f}, U: {dict2["U"]:.5f}, H: {dict2["H"]:.5f}, CZSL: {dict2["CZSL"]:.5f}')

        logger.info(f'Epoch: {dict1["epoch"]}')
        logger.info("Performance => S:{:.5f}; U:{:.5f}; H:{:.5f}; CZSL:{:.5f}; Acc_train:{:.5f}".format(
            dict1['S'], dict1['U'], dict1['H'], dict1['CZSL'], dict1['acc_train']))
        logger.info('Best GZSL|CZSL => S:{:.5f}; U:{:.5f}; H:{:.5f}; CZSL:{:.5f}'.format(
            best_performance[0], best_performance[1], best_performance[2], best_performance[3]))
        logger.info("-"*50)
        loss_record = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print("---------- End ----------")


if __name__ == "__main__":
    args = parse_args()
    print('args:', args)
    if args.is_end2end:
        print("==> End to end mode")
        end2end_train(args)
    else:
        print("==> Use extracted feature")
        train(args)