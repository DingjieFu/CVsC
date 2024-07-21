import os
import argparse


def parse_args():
    projectPath = os.path.dirname(os.path.abspath(__file__))
    datarootPath = projectPath + "/data"
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--projectPath', default=projectPath, help='')
    parser.add_argument('--device', default='cuda:0', help='cpu/cuda:x')
    parser.add_argument('--seed', default=2024, type=int, help='seed for reproducibility')
    parser.add_argument('--save', default=True, help='Save the trained model or not')
    parser.add_argument('--is_end2end', default=False, action="store_true", help='End to end mode')
    parser.add_argument('--log_root_path', default=projectPath + '/out', help='Save path of exps')

    # -------------------- Dataset config --------------------#
    parser.add_argument('--dataset', default='AWA2', help='dataset: AWA2/CUB/SUN')
    parser.add_argument('--image_root', default= datarootPath + '/data/dataset', help='Path to image root')
    parser.add_argument('--mat_path', default= datarootPath + '/data/dataset/xlsa17/data',
                        help='Features extracted from pre-training Resnet')
    parser.add_argument('--attr_path', default= datarootPath + '/data/attribute', help='attribute path')

    # -------------------- train config --------------------#
    parser.add_argument('--backbone', default='ViT', help='ViT/Resnet')
    parser.add_argument('--inputsize', type=int, default=224, help='input_imgage_size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='Input batch size')
    parser.add_argument('--nepoch', type=int, default=20, help='Number of epochs to train')

    # -------------------- DAZLE config --------------------#
    parser.add_argument('--w2v_att', default=True, help='use word vector or not')
    parser.add_argument('--w2v_path', default= datarootPath + '/data/w2v', help='w2v path')
    parser.add_argument('--trainable_w2v', default=True, help='word vector trainable or not')

    # -------------------- hyper-params --------------------#
    parser.add_argument('--alpha0', type=float, default=3.0, help='efficents of CrossEntropy loss')
    parser.add_argument('--alpha1', type=float, default=0.4, help='efficents of Calibrate loss')
    parser.add_argument('--alpha2', type=float, default=0.0, help='efficents of CCL')
    parser.add_argument('--alpha3', type=float, default=0.0, help='efficents of SCI')
    parser.add_argument('--alpha4', type=float, default=0.0, help='efficents of TCI')

    # -------------------- zero-shot config --------------------#
    # bias fator for unseen classes
    parser.add_argument('--bias', type=float, default=2.0, help='')
    parser.add_argument('--is_bias', default=True, help='is_bias or not')

    # -------------------- dataloader config --------------------#
    parser.add_argument('--is_scale', default=False, help='')
    parser.add_argument('--is_balance', default=False, help='class balance')
    parser.add_argument('--is_pairs', default=False, help='input sample pairwise')

    # for distributed loader
    parser.add_argument('--train_mode', type=str, default='random', help='loader: random or distributed')
    parser.add_argument('--n_batch', type=int, default=1000, help='batch numbers per epoch')
    parser.add_argument('--ways', type=int, default=16, help='class numbers per episode')
    parser.add_argument('--shots', type=int, default=2, help='image numbers per class')

    args = parser.parse_args()
    return args
