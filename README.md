# EVA-GCN: Head Pose Estimation Based on Graph Convolutional Networks

**[accepted to 2021 CVPR Workshop]**

![alt text](https://github.com/stoneMo/EVA-GCN/blob/main/imgs/title_image.png?raw=true)

## Abstract

Head pose estimation is an important task in many real-world applications. Since the facial landmarks usually serve as the common input that is shared by multi- ple downstream tasks, utilizing landmarks to acquire high- precision head pose estimation is of practical value for many real-world applications. However, existing landmark-based methods have a major drawback in model expressive power, making them hard to achieve comparable performance to the landmark-free methods. In this paper, we propose a strong baseline method which views the head pose estimation as a graph regression problem. We construct a landmark-connection graph, and propose to lever- age the Graph Convolutional Networks (GCN) to model the complex nonlinear mappings between the graph typologies and the head pose angles. Specifically, we design a novel GCN architecture which utilizes joint Edge-Vertex Attention (EVA) mechanism to overcome the unstable landmark detection. Moreover, we introduce the Adaptive Channel Attention (ACA) and the Densely-Connected Architecture (DCA) to boost the performance further. We evaluate the proposed method on three challenging benchmark datasets. Experiment results demonstrate that our method achieves better performance in comparison with the state-of-the-art landmark-based and landmark-free methods.



## Requirements


To install requirements, you can:

```
pip install -r requirements.txt

```

Note that our implementation is based on Python 3.7, and PyTorch deep learning framework, trained on NVIDIA Titan RTX GPU in Ubuntu 16.04 system.

## Codes

There are three different section of this project. 
1. Data pre-processing
2. Training and testing 

We will go through the details in the following sections.

### 1. Data pre-processing

In this work, we demonstrate the effective of the proposed OsGG-Net for head pose estimation on BIWI Kinect Head Pose Database. The BIWI dataset contains 24 videos of 20 subjects in the controlled environment. There are a total of roughly 15, 000 frames in the dataset. The 300W across Large Poses (300W-LP) dataset is synthesized with expanding 558 61,225 samples across large poses in the 300W dataset with flipping to 122,450 samples. The AFLW2000 dataset provides 560 ground-truth 3D faces and the corresponding 68 landmarks for the first 2, 000 images of the AFLW dataset, where the faces in the dataset have large pose variations with various illumination conditions and expressions. 


```

# Running on BIWI dataset

cd data_preprocessing
python create_db_biwi.py

# Running on AFLW dataset

cd data_preprocessing
bash run_created_db_300W_AFLW.bash 

```

If you don't want to re-download every dataset images and do the pre-processing again, or maybe you don't even care about the data structure in the folder. Just download the file **data.zip** from the following link, and replace the data subfolder in the data_preprocessing folder.

[Google drive](https://drive.google.com/file/d/1myA0xpN0t1Zkufz7Ptk6zmIkp1XFYB0S/view?usp=sharing)

Now you can skip to the "Training and testing" stage. If you want to do the data pre-processing from the beginning, you need to download the dataset first, and unzip the dataset in the dataset folder for data_preprocessing.

#### Download the datasets

+ [BIWI Kinect Head Pose Database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)
+ [300W-LP+AFLW](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)


### 2. Training and testing 
```

# Running on BIWI dataset

sh run_train_on_300W_LP_test_on_BIWI.sh


# Running on AFLW dataset

sh run_train_on_300W_LP_test_on_AFLW.sh

```

Just remember to check which dataset type you want to use in the shell and you are good to go. Note that we calculate the MAE of yaw, pitch, roll independently, and average them into one single MAE for evaluation.

