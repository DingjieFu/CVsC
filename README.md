<div align="center">
<h1> CVsC [ACM MM'24]🎉 </h1>
<h3> Causal Visual-semantic Correlation for Zero-shot Learning </h3>

[Shuhuang Chen]()<sup>1 *︎</sup> ,[Dingjie Fu](https://github.com/DingjieFu)<sup>1 *︎</sup> ,[Shiming Chen](https://shiming-chen.github.io/)<sup>2</sup> ,[Shuo Ye](https://github.com/SYe-hub)<sup>1</sup> ,[Wenjin Hou](https://github.com/Houwenjin)<sup>1</sup> ,[Xinge You](https://bmal.hust.edu.cn/EN.htm)<sup>1 ✉</sup>

<sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> Mohamed bin Zayed University of AI 
<br>
(*︎) co-first author,  (✉) corresponding author
<br>
</div>


## Overview
<details>
## Introduction


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
python train.py --dataset CUB --backbone Resnet --inputsize 224 # after finishing extracting features
```

**End to end training**
```
python train.py --dataset CUB --backbone Resnet --inputsize 224 --is_end2end
```
## 📑Acknowledgement
The implementation is based on the repos: [DAZLE]([https://github.com/dyh127/RFNet](https://github.com/hbdat/cvpr20_DAZLE)), thanks for their excellent works.


