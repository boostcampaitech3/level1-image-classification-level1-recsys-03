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

# 2022.02.28 (fix augmentation)
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK --label mask

# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --model_dir ./model/age/ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE ./model/gender/ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER ./model/mask/ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK
# python inference.py --model VGGFace --model_param resnet false --label age gender mask --model_dir ./model/age/VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_AGE ./model/gender/VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_GENDER ./model/mask/VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_MASK

# compare weight version diff
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv3_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv3_MASK --label mask

# python inference.py --model VGGFace --model_param false --label age gender mask --output_filename output-VGGFace_Ep60_Weightv3_AGM-20220301-ijkimmmy.csv --model_dir ./model/age/VGGFace_Ep60_Weightv3_AGE ./model/gender/VGGFace_Ep60_Weightv3_GENDER ./model/mask/VGGFace_Ep60_Weightv3_MASK

# efficientnet try1
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 1 --name VGGFace_Ep60_Weightv1_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 1 --name VGGFace_Ep60_Weightv1_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 1 --name VGGFace_Ep60_Weightv1_MASK --label mask

# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --name Effb3_Ep60_Weightv0_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --name Effb3_Ep60_Weightv0_GENDER --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --name Effb3_Ep60_Weightv0_MASK --label mask

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 2 --name VGGFace_Ep60_Weightv2_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 2 --name VGGFace_Ep60_Weightv2_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 2 --name VGGFace_Ep60_Weightv2_MASK --label mask

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 0 --name VGGFace_Ep60_Weightv0_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 0 --name VGGFace_Ep60_Weightv0_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 0 --name VGGFace_Ep60_Weightv0_MASK --label mask

# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep60_Weightv0_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep60_Weightv0_GENDER --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep60_Weightv0_MASK --label mask

# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 3 --name Vgg_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 2 --name Vgg_Ep60_Weightv2_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 1 --name Vgg_Ep60_Weightv1_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 4 --name Vgg_Ep60_Weightv4_AGE --label age

# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --lr 0.01 --weight_version 3 --name Effb3_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 4 --name Vgg_Ep60_Weightv4_GENDER --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 4 --name Vgg_Ep60_Weightv4_MASK --label mask

# python train.py --epochs 60 --model VGGFace --model_param true --optimizer SGD --weight_version 3 --name VGGFace_Feature_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --model VGGFace --model_param true --optimizer SGD --weight_version 3 --name VGGFace_Feature_Ep60_Weightv3_GENDER --label gender
# python train.py --epochs 60 --model VGGFace --model_param true --optimizer SGD --weight_version 3 --name VGGFace_Feature_Ep60_Weightv3_MASK --label mask

# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 2 --name Vgg_Ep60_Weightv2_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --optimizer SGD --weight_version 3 --name ResNet_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 4 --name Vgg_Ep60_Weightv4_AGE --label age

# python train.py --epochs 60 --model VGGFace --model_param true --optimizer SGD --weight_version 1 --name VGGFace_Feature_Ep60_Weightv1_AGE --label age
# python inference.py --model VGGFace --model_param false --output_filename output-VGGFace_Feature_Ep60_Weightv1_AGM-20220301-ijkimmmy.csv --label age gender mask --model_dir ./model/age/VGGFace_Feature_Ep60_Weightv1_AGE ./model/gender/VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_GENDER ./model/mask/VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_MASK

# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --weight_version 1 --name ResNet50_Ep60_Weightv1_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --weight_version 1 --name ResNet50_Ep60_Weightv1_GENDER --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --weight_version 1 --name ResNet50_Ep60_Weightv1_MASK --label mask
# python inference.py --model PretrainedModels --model_param resnet false --output_filename output-ResNet50_Ep60_Weightv1_AGM-20220301-ijkimmmy.csv --label age gender mask --model_dir ./model/age/ResNet50_Ep60_Weightv1_AGE ./model/gender/ResNet50_Ep60_Weightv1_GENDER ./model/mask/ResNet50_Ep60_Weightv1_MASK

# k fold implemented # ResNet50_Ep60_Weightv0_AGE5
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --weight_version 0 --name ResNet50_Ep60_Weightv0_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --weight_version 0 --name Effb3_Ep60_Weightv0_AGE2 --label age
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --weight_version 0 --name ResNet50_Ep60_Weightv0_GENDER --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --weight_version 0 --name ResNet50_Ep60_Weightv0_MASK --label mask

# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --weight_version 0 --name ResNet50_Ep60_Weightv0_GENDER2 --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --weight_version 0 --name ResNet50_Ep60_Weightv0_MASK2 --label mask

# python inference.py --model PretrainedModels --model_param resnet false --output_filename output-ResNet50_kfold_Ep60_Weightv0_AGM-20220302-ijkimmmy.csv --label age gender mask --model_dir ./model/age/ResNet50_Ep60_Weightv0_AGE ./model/gender/ResNet50_Ep60_Weightv0_GENDER2 ./model/mask/ResNet50_Ep60_Weightv0_MASK2

# confusion matrix & count incorrect labels in train set
# python train.py --epochs 60 --model VGGFace --model_param true --weight_version 0 --name VGGFace_Feature_Ep60_Weightv0_AGE2 --label age
# python train.py --epochs 60 --model VGGFace --model_param false --weight_version 0 --name VGGFace_Ep60_Weightv0_AGE2 --label age
# test weight (weight3 >>>> weight0)
# python train.py --epochs 60 --model PretrainedModels --model_param resnet50 false --weight_version 3 --name ResNet50_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param resnet50 false --weight_version 3 --name ResNet50_Ep60_Weightv3_GENDER --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param resnet50 false --weight_version 3 --name ResNet50_Ep60_Weightv3_MASK --label mask
# python inference.py --model PretrainedModels --model_param resnet false --output_filename output-ResNet50_kfold_Ep60_Weightv3_AGM-20220302-ijkimmmy.csv --label age gender mask --model_dir ./model/age/ResNet50_Ep60_Weightv3_AGE ./model/gender/ResNet50_Ep60_Weightv3_GENDER ./model/mask/ResNet50_Ep60_Weightv3_MASK

# test ResNet18 vs ResNet50
# python train.py --epochs 60 --model PretrainedModels --model_param resnet18 false --weight_version 3 --name ResNet18_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param resnet18 false --weight_version 3 --name ResNet18_Ep60_Weightv3_GENDER --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param resnet18 false --weight_version 3 --name ResNet18_Ep60_Weightv3_MASK --label mask
# python inference.py --model PretrainedModels --model_param resnet18 false --output_filename output-ResNet18_kfold_Ep60_Weightv3_AGM-20220302-ijkimmmy-v13.csv --label age gender mask --model_dir ./model/age/ResNet18_Ep60_Weightv3_AGE ./model/gender/ResNet18_Ep60_Weightv3_GENDER ./model/mask/ResNet18_Ep60_Weightv3_MASK

# python inference_copy.py --output_filename output-Voting-20220302-ijkimmy-v14.csv --model_dirs \
# ./model/age/ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_AGE ./model/gender/ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_GENDER ./model/mask/ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_MASK \
# ./model/age/ResNet50_Ep60_Weightv0_AGE ./model/gender/ResNet50_Ep60_Weightv0_GENDER2 ./model/mask/ResNet50_Ep60_Weightv0_MASK2 \
# ./model/age/ResNet50_Ep60_Weightv3_AGE ./model/gender/ResNet50_Ep60_Weightv3_GENDER ./model/mask/ResNet50_Ep60_Weightv3_MASK \
# ./model/age/ResNet18_Ep60_Weightv3_AGE ./model/gender/ResNet18_Ep60_Weightv3_GENDER ./model/mask/ResNet18_Ep60_Weightv3_MASK

# test vggface finetune
# python train.py --epochs 60 --model VGGFace --model_param true --weight_version 3 --name VGGFace_Feature_kf_Ep60_Weightv3_AGE --label age
# wrong (didn't freeze layers)
# python train.py --epochs 10 --model VGGFace --model_param true --weight_version 3 --name VGGFace_Feature_kf_Ep10_Weightv3_AGE --label age
# python train.py --epochs 10 --model VGGFace --model_param true --weight_version 3 --name VGGFace_Feature_kf_Ep10_Weightv3_GENDER --label gender
# python train.py --epochs 10 --model VGGFace --model_param true --weight_version 3 --name VGGFace_Feature_kf_Ep10_Weightv3_MASK --label mask
# python inference.py --model VGGFace --model_param true --output_filename output-VGGFace_Ep10_AGM-20220303-ijkimmmy-v13.csv --label age gender mask --model_dir ./model/age/VGGFace_Feature_kf_Ep10_Weightv3_AGE ./model/gender/VGGFace_Feature_kf_Ep10_Weightv3_GENDER ./model/mask/VGGFace_Feature_kf_Ep10_Weightv3_MASK

# python train.py --epochs 10 --model VGGFace --model_param false --weight_version 3 --name VGGFace_kf_Ep10_Weightv3_AGE --label age
# python train.py --epochs 10 --model VGGFace --model_param false --weight_version 3 --name VGGFace_kf_Ep10_Weightv3_GENDER --label gender
# python train.py --epochs 10 --model VGGFace --model_param false --weight_version 3 --name VGGFace_kf_Ep10_Weightv3_MASK --label mask

python train.py --epochs 12 --model VGGFace --model_param true --weight_version 3 --name VGGFace_Feature_kf_Ep12_Weightv3_AGE --label age
python train.py --epochs 12 --model VGGFace --model_param true --weight_version 3 --name VGGFace_Feature_kf_Ep12_Weightv3_GENDER --label gender
python train.py --epochs 12 --model VGGFace --model_param true --weight_version 3 --name VGGFace_Feature_kf_Ep12_Weightv3_MASK --label mask
python inference.py --model VGGFace --model_param true --output_filename output-VGGFace_Feature_kf_Ep12_AGM-20220303-ijkimmmy-v14.csv --label age gender mask --model_dir ./model/age/VGGFace_Feature_kf_Ep12_Weightv3_AGE ./model/gender/VGGFace_Feature_kf_Ep12_Weightv3_GENDER ./model/mask/VGGFace_Feature_kf_Ep12_Weightv3_MASK


