<div align="center">
<h1> CVsC [ACM MM'24]🎉 </h1>
<h3> Causal Visual-semantic Correlation for Zero-shot Learning </h3>

[Shuhuang Chen](https://github.com/shchen0001)<sup>1</sup> ,[Dingjie Fu](https://github.com/DingjieFu)<sup>1</sup> ,[Shiming Chen](https://shiming-chen.github.io/)<sup>2</sup> ,[Shuo Ye](https://github.com/SYe-hub)<sup>1</sup> ,[Wenjin Hou](https://github.com/Houwenjin)<sup>3</sup>

<sup>1</sup> Huazhong University of Science and Technology

<sup>2</sup> Mohamed bin Zayed University of AI, <sup>3</sup> ReLER Lab, Zhejiang University, China
<br>
<br>
</div>


## 🤓Overview
<details>

### • Abstract
Zero-Shot learning (ZSL) correlates visual samples and shared semantic information to transfer knowledge from seen classes to unseen classes. 
Existing methods typically establish visual-semantic correlation by aligning visual and semantic features, which are extracted from visual samples and semantic information, respectively. 
However, instance-level images, owing to singular observation perspectives and diverse individuals, cannot exactly match the comprehensive semantic information defined at the class level. 
Direct feature alignment imposes correlation between mismatched vision and semantics, resulting in spurious visual-semantic correlation. 
To address this, we propose a novel method termed Causal Visualsemantic Correlation (CVsC) to learn substantive visual-semantic correlation for ZSL. 
Specifically, we utilize a Visual Semantic Attention module to facilitate interaction between vision and semantics, thereby identifying attribute-related visual features. 
Furthermore, we design a Conditional Correlation Loss to properly utilize semantic information as supervision for establishing visual-semantic correlation. 
Moreover, we introduce counterfactual intervention applied to attribute-related visual features, and maximize their impact on semantic and target predictions to enhance substantive visual-semantic correlation. 
Extensive experiments conducted on three benchmark datasets (i.e., CUB, SUN, and AWA2) demonstrate that our CVSC outperforms existing state-of-the-art methods.

<div align="center"><img src="assets/motivation.png" /></div>

### • Framework
<div align="center"><img src="assets/framework.png" /></div>


### • Main Results
| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 79.1 | 72.4 | 78.4 | 75.3 |
| SUN | 71.5 | 61.9 | 47.6 | 53.8 |
| AWA2 | 73.1 | 68.0 | 87.0 | 76.4 |

</details>


## 💪Getting Started
<h3> • Requirements </h3>

```
git clone git@github.com:DingjieFu/CVsC.git
cd CVsC
conda create -n CVsC python=3.8.18
conda activate CVsC
pip install -r requirements.txt
```
<h3> • Data Preparation </h3>

🌟 **Note: You can download the datasets by following the instructions in the '[dataset](https://github.com/DingjieFu/CVsC/tree/main/data/dataset)' directory**
```
CVsC/
├── data
│   ├── attribute
│   ├── dataset
│   │   ├── AWA2
│   │   │   ├── Animals_with_Attributes2
│   │   │   │   ├── JPEGImages
│   │   │   │   └── ...
│   │   ├── CUB
│   │   │   ├── CUB_200_2011
│   │   │   │   ├── images
│   │   │   │   └── ...
│   │   ├── SUN
│   │   │   ├── images
│   │   │   └── ...
│   │   ├── xlsa
│   │   │   ├── data
│   │   │   │   ├── AWA2
│   │   │   │   └── ...
│   ├── w2v
│   └── ...
├── extract_feature
└── ...
```
<h3> • Model Training </h3>

**Using extracted features**
```
python extract_feature/extract_feature_map_ResNet_101.py --dataset CUB --inputsize 224 --batch_size 500
python train.py --dataset CUB --backbone Resnet --inputsize 224 # after extracting features
```

**End to end training**
```
python train.py --dataset CUB --backbone Resnet --inputsize 224 --is_end2end
```
## 📑Acknowledgement
The implementation is based on the repo [DAZLE](https://github.com/hbdat/cvpr20_DAZLE), thanks for their excellent works.

## ✨Citation
If you find CVsC is useful in your research or applications, welcome to cite our paper and give us a star 🌟.
```bibtex
@inproceedings{chen2024causal,
  title={Causal Visual-semantic Correlation for Zero-shot Learning},
  author={Chen, Shuhuang and Fu, Dingjie and Chen, Shiming and Ye, Shuo and Hou, Wenjin and You, Xinge},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={4246--4255},
  year={2024}
}
