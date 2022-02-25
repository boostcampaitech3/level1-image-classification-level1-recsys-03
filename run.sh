# 2022-02-25
# separate label & implemented WeightedRandomSample
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug
python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug_AGE --label age
