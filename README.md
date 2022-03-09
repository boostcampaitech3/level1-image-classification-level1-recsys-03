# Boostcamp AI Tech3 Image Clasification Contest (Level 1)
---

## Contents
- [프로젝트 개요](#-1.-프로젝트-개요)
- [팀원 소개](#-2.-프로젝트-팀-구성-및-역할)
- [Ground Rule](#-3.-Ground-Rule)
- [프로젝트 구조](#-4.-프로젝트-구조)
- [Getting Started](#-5.-Getting-Started)

# 1. 프로젝트 개요

## 프로젝트 주제

<aside>
🧑🏻‍💻 COVID-19 확산 방지를 위해 중요한 것은 올바른 마스크 착용이다. 이를 검사하는 인적자원을 최소화하기 위해 카메라로 비춰진 얼굴 이미지 만으로 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 Image Classification 기반으로 구분할 수 있는 시스템을 구축하려고 한다.

</aside>

## 데이터셋 구성

- 아시아인 남녀로 구성되었다.
- 연령대는 20대부터 70대까지 다양하게 분포하고 있다.
- 전체 4500명 = train set 2700명 (60%) + test set 1800명 (40%)
- 한 명당 7장 = no mask 1장 + incorrect mask 1장 + correct mask 5장
- 한 장당 512*384의 사이즈를 가진다.
- 현재 task에서의 class

    ![class_labels](/class_labels.png)

## 개발환경

- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- Ram 90GB
- Tesla V100 32GB


# 2. 프로젝트 팀 구성 및 역할

- 박정훈 (iksadNorth)
    - confusion matrix 사용한 실험 결과 시각화 구현
    - 중복되는 Mask data undersampling 구현
- 김인정 (ijkimmy)
    - ImageNet 기학습 모델들과 VGGFace 성능 비교 (+ fine-tuning)
    - RandomWeightedSampler, WeightedCrossEntropyLoss, StratifiedKFold, Early Stopping, Ensemble (hard, soft voting) 구현
- 진상우 (Jin)
    - ViT 기반 모델 & optimizer (Adam, AdamW) 실험
- 김진우 (jinu)
    - wandb sweep 사용한 hyperparameter tuning 실험
- 서정빈 (jeongbeen)
    - Focal, Label smoothing, Arcface loss function 실험
    - Learning rate scheduler 실험 (StepLR, ExponentialLR)

# 3. Ground Rule

## Coding Convention

- Naming
    - 누구나 알 수 있는 **쉽고 직관적인** 단어 사용
        
        
        | namespace | notation |
        | --- | --- |
        | class | UpperCamelCase |
        | function | snake_case |
        | variable | snake_case |
- 클래스를 **기능 별로 세분화**하여 최대한 작게 만든다.
- 반복되는 코드를 작성하지 않는다.
- **가독성**이 좋은 코드를 작성하자.

## Organization

- 원활한 의사소통을 위해 Slack과 zoom을 활용
- 실험 공유: tensorboard와 wandb를 활용


# 4. 프로젝트 구조

```python
ijkimmy/
├── model/                  # a default directory for saving model output
│  └── loss.py              # loss function classes (e.g focal loss, label smoothing loss, f1 loss)
│  └── model.py             # model class that inherits nn.Module (e.g. PretrainedModels, VGGFace)
│
├── output/                 # a default directory for inference result files 
│
├── project_reports/        # include a review and a report about project timeline and 
│
├── utils/                  # small utility functions
│  └── util.py	            # has functions like EarlyStopping etc.
│
├── data_viz.ipynb          # evaluate model using confusion matrix
│
├── dataset.py              # dataset class that inherits torch.utils.data.Dataset
│
├── inference.py            # inference from trained model (make inference result to csv form)
│
├── run.sh                  # script to run the project 
│
└── train.py                # setting and implementation of training
```


# 5. Getting Started    
## Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

## Install Requirements
- `pip install -r requirements.txt`

## Run
- Run the program by modifying the `run.sh` script. By default, it trains three separate ResNet18 models to classify age, gender, and mask using `train.py`, and then creates an output file using `inference.py`.

## Acknowledgements
This project is generated from the template [Pytorch-Template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque)
