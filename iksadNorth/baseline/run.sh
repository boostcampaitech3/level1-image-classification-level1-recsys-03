# 2022.02.26
python train.py --epochs 50 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer SGD --name VGGFace_Ep20_SplitProf_CustAugv1_WeightedCE_SGD

# 2022.02.27
# 비교군 <loss 비교> [--criterion cross_entropy] -> [--criterion F1Loss] -> [--criterion label_smoothing]
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model ResNetModel --optimizer SGD --name UniformDataset_BaseAugmentation_ResNetModel_SGD
### 단순히 loss.py의 f1을 그대로 사용하였다.(F1Loss의 클래스의 init파라미터 classes=18로 지정)
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model ResNetModel --optimizer SGD --criterion f1 --name UniformDataset_BaseAugmentation_ResNetModel_SGD_F1Loss
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model ResNetModel --optimizer SGD --criterion label_smoothing --name UniformDataset_BaseAugmentation_ResNetModel_SGD_label_smoothing

# 비교군 <loss 비교> [--dataset UniformDataset] -> [--dataset UniGakGakDataset]
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model ResNetModel --optimizer SGD --name UniformDataset_BaseAugmentation_ResNetModel_SGD

# UniGakGakDataset_BaseAugmentation_ResNetModel_SGD_age
python train.py --epochs 50 --dataset UniGakGakDataset --augmentation BaseAugmentation --model ResNetModel --optimizer SGD --label_type age --name UniGakGakDataset_BaseAugmentation_ResNetModel_SGD
# UniGakGakDataset_BaseAugmentation_ResNetModel_SGD_gender
python train.py --epochs 50 --dataset UniGakGakDataset --augmentation BaseAugmentation --model ResNetModel --optimizer SGD --label_type gender --name UniGakGakDataset_BaseAugmentation_ResNetModel_SGD
# UniGakGakDataset_BaseAugmentation_ResNetModel_SGD_mask
python train.py --epochs 50 --dataset UniGakGakDataset --augmentation BaseAugmentation --model ResNetModel --optimizer SGD --label_type mask --name UniGakGakDataset_BaseAugmentation_ResNetModel_SGD

# 비교군 <loss 비교> [--dataset UniformDataset] -> [--dataset UniGakGakDataset]
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model ResNetModel --optimizer SGD --name UniformDataset_BaseAugmentation_ResNetModel_SGD

# 실험 번호 001 : UniformDataset_BaseAugmentation_PretrainedModel('resnet18', freeze)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model resnet18 --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-resnet18-_SGD
# 실험 번호 001 : UniformDataset_BaseAugmentation_PretrainedModel('resnet18', melt)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model resnet18 false --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-resnet18-melt-_SGD
# 실험 번호 001 : UniformDataset_BaseAugmentation_PretrainedModel('vgg11', freeze)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model vgg11 true --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-vgg11-_SGD
# 실험 번호 001(실패 호환 불가함.) : UniformDataset_BaseAugmentation_PretrainedModel('inception_v3', freeze)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model inception_v3 true --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-inception_v3-_SGD
# 실험 번호 001 : UniformDataset_BaseAugmentation_PretrainedModel('squeezenet1_0', freeze)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model squeezenet1_0 true --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-squeezenet1_0-_SGD
# 실험 번호 001 : UniformDataset_BaseAugmentation_PretrainedModel('alexnet', freeze)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model alexnet true --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-alexnet-_SGD

# 실험 번호 001 : UniformDataset_BaseAugmentation_PretrainedModel('efficientnet_b0', freeze)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model efficientnet_b0 true --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-efficientnet_b0-_SGD
# 실험 번호 001 : UniformDataset_BaseAugmentation_PretrainedModel('densenet121', freeze)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model densenet121 true --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-densenet121-_SGD
# 실험 번호 001 : UniformDataset_BaseAugmentation_PretrainedModel('vit_b_16', freeze)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model vit_b_16 true --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-vit_b_16-_SGD
# 실험 번호 001 : UniformDataset_BaseAugmentation_PretrainedModel('efficientnet_b7', freeze)_SGD
python train.py --epochs 50 --dataset UniformDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model efficientnet_b7 true --optimizer SGD --name UniformDataset_BaseAugmentation_PretrainedModel-efficientnet_b7-_SGD
