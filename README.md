# Few-Shot Defect Classification via Feature Aggregation Based on Graph Neural Network
![image](https://github.com/Harry10459/CIDnet/assets/90323751/3bafeda0-bd4c-42b5-a6cb-f886c355d3de)

# Abstract
The effectiveness of deep learning models is greatly dependent on the availability of a vast amount of labeled data. However, in the realm of surface defect classification, acquiring and annotating defect samples proves to be quite challenging. Consequently, accurately predicting defect types with only a limited number of labeled samples has emerged as a prominent research focus in recent years. Few-shot learning, which leverages a restricted sample set in the support set, can effectively predict the categories of unlabeled samples in the query set. This
approach is especially well-suited for the context of defect classification. In this article, we propose a novel few-shot surface defect classification method, which using both the instance-level relations and distribution-level relations in each few-shot learning task. Furthermore, we propose incorporating class center features into the feature aggregation operation to rectify the positioning of edge samples in the mapping space. This adjustment aims to minimize the distance between samples of the same category, thereby mitigating the influence on the classification of unlabeled images at the category boundary. Experimental results on the public dataset show the outstanding performance of our proposed approach compared to the state-of-the-art methods in the few-shot learning settings.

# Dataset
MSD-Cls dataset:
link：https://pan.baidu.com/s/14-x_blzNvtY7N5Ue1U2skw password：z584

DGAM 2007 dataset:
link: https://aistudio.baidu.com/aistudio/datasetdetail/97571

# Training
```python  main_proto.py --dataset_root dataset --config config/proto_5way_5shot_resnet12_msd.py --num_gpu 2 --mode train```

# Evaluation
```python  main_proto.py --dataset_root dataset --config config/proto_5way_5shot_resnet12_msd.py --num_gpu 2 --mode eval```
