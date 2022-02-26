# 2022-02-25
# separate label & implemented WeightedRandomSample
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug_AGE --label age
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug_MASK --label mask
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug_GENDER --label gender

# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAug
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAug_AGE --label age
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAug_MASK --label mask
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAug_GENDER --label gender

# python train.py --epochs 30 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAugv1

# python inference.py --model VGGFace --model_dir ./model/multi/VGGFace_Ep20_SplitProf_CustAugv1
# python inference.py --model VGGFace --label age gender mask --model_dir ./model/age/VGGFace_Ep20_SplitProf_CustAug_AGE ./model/gender/VGGFace_Ep20_SplitProf_CustAug_GENDER ./model/mask/VGGFace_Ep20_SplitProf_CustAug_MASK

# python train.py --epochs 50 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer Adam --name VGGFace_Ep20_SplitProf_CustAugv1_Adam
# python train.py --epochs 50 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer SGD --name VGGFace_Ep20_SplitProf_CustAugv1_WeightedCE_SGD

# 2022.02.26
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD
# python inference.py --model PretrainedModels --model_param resnet false --model_dir ./model/multi/ResNet_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD

python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_SplitProf_CustAugv1_WeightedCE_ImbalanceSample_SGD