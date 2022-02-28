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
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_SplitProf_CustAugv1_WeightedCEnNormSample_SGD
# python inference.py --model PretrainedModels --model_param resnet false --model_dir ./model/multi/ResNet_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet true --optimizer SGD --name ResNet_Feature_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Freeze6_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_fcxavier_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --name ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --name ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD_MASK --label mask
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --name ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD_GENDER --label gender

# 2022.02.27
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param alexnet false --optimizer SGD --name Alexnet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param squeezenet false --optimizer SGD --name Squeezenet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param densenet false --optimizer SGD --name Densenet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param inception false --optimizer SGD --name Inception_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Feature_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_AGE --label age
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_GENDER --label gender
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_MASK --label mask

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer SGD --name VGGFace_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_AGE --label age
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer SGD --name VGGFace_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_MASK --label mask
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer SGD --name VGGFace_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_GENDER --label gender

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience30_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_AGE --label age
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience30_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_GENDER --label gender
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience30_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_MASK --label mask
# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --model_dir ./model/age/ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_AGE ./model/gender/ResNet_Ep120_patience30_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_GENDER ./model/mask/ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD_MASK

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_AGE --label age
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_GENDER --label gender
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_MASK --label mask
# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --model_dir ./model/age/ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_AGE ./model/gender/ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_GENDER ./model/mask/ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_MASK

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_MASK --label mask

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param true --optimizer SGD --name VGGFace_Feature_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_AGE --label age

# 2022.02.28
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE --label age
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER --label gender
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK --label mask

python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet true --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE --label age
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet true --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER --label gender
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet true --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK --label mask

python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE --label age
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER --label gender
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK --label mask

python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param true --optimizer SGD --name VGGFace_Feature_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE --label age
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param true --optimizer SGD --name VGGFace_Feature_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER --label gender
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param true --optimizer SGD --name VGGFace_Feature_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK --label mask

python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE --label age
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER --label gender
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK --label mask

python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param vgg true --optimizer SGD --name Vgg_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE --label age
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param vgg true --optimizer SGD --name Vgg_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER --label gender
python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param vgg true --optimizer SGD --name Vgg_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK --label mask