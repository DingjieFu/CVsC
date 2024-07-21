import os
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import gensim.downloader as api
from gensim.models import KeyedVectors
warnings.filterwarnings("ignore")
fileDirPath = os.path.dirname(__file__)
fileDirName = os.path.basename(fileDirPath)
rootPath = fileDirPath.replace(fileDirName, "")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract attribute w2v")
    parser.add_argument('--dataset', default='CUB', help='dataset: AWA2/CUB/SUN')
    parser.add_argument('--attr_path', default= rootPath + 'data/attribute', help='attribute path')
    parser.add_argument('--w2v_path', default= rootPath + 'data/w2v/', help='w2v path')
    args = parser.parse_args()
    return args
args = parse_args()


print('Loading pretrain w2v model')
custom_save_path = fileDirPath + "/pretrainedModel/word2vec-google-news-300.model"
if os.path.exists(custom_save_path):
    model = KeyedVectors.load(custom_save_path)
else:
    model_name = 'word2vec-google-news-300' # best model
    model = api.load(model_name)
    os.makedirs(os.path.dirname(custom_save_path), exist_ok=True)
    custom_save_path = fileDirPath + "/pretrainedModel/word2vec-google-news-300.model"
    model.save(custom_save_path)
dim_w2v = 300
print('Done loading model')


if args.dataset == "AWA2":
    replace_word = [('newworld','new world'),('oldworld','old world'),('nestspot','nest spot'),('toughskin','tough skin'),
                ('longleg','long leg'),('chewteeth','chew teeth'),('meatteeth','meat teeth'),('strainteeth','strain teeth'),
                ('quadrapedal','quadrupedal')]
    path = args.attr_path + f'/{args.dataset}/predicates.txt'
    df=pd.read_csv(path,sep='\t',header = None, names = ['idx','des'])
    new_des = df['des'].values

elif args.dataset == "CUB":
    replace_word = [('spatulate','broad'),('upperparts','upper parts'),('grey','gray')]
    path = args.attr_path + f'/{args.dataset}/attributes.txt'
    df = pd.read_csv(path,sep=' ',header = None, names = ['idx','des'])
    des = df['des'].values
    new_des = [' '.join(i.split('_')) for i in des]
    new_des = [' '.join(i.split('-')) for i in new_des]
    new_des = [' '.join(i.split('::')) for i in new_des]
    new_des = [i.split('(')[0] for i in new_des]
    new_des = [i[4:] for i in new_des]

elif args.dataset == "SUN":
    replace_word = [('rockstone','rock stone'),('dirtsoil','dirt soil'),('man-made','man-made'),('sunsunny','sun sunny'),
                ('electricindoor','electric indoor'),('semi-enclosed','semi enclosed'),('far-away','faraway')]
    file_path = args.attr_path + f'/{args.dataset}/attributes.mat'
    matcontent = sio.loadmat(file_path)
    des = matcontent['attributes'].flatten()
    df = pd.DataFrame()
    new_des = [''.join(i.item().split('/')) for i in des]
else:
    print("Unkonwn Dataset!")
    exit()


# replace out of dictionary words
for pair in replace_word:
    for idx,s in enumerate(new_des):
        new_des[idx]=s.replace(pair[0],pair[1])
print('Done replace OOD words')
df['new_des'] = new_des
df.to_csv(args.attr_path + f'/{args.dataset}/new_des.csv')
print('Done preprocessing attribute des')
all_w2v = []
for s in new_des:
    print(s)
    words = s.split(' ')
    if words[-1] == '':     # remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
    all_w2v.append(w2v[np.newaxis,:])
all_w2v=np.concatenate(all_w2v,axis=0)
print(all_w2v.shape)
with open(args.w2v_path + f'/{args.dataset}_attribute_bert.pkl','wb') as f:
    pickle.dump(all_w2v,f)