# output_UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_SGD(0.1382	31.6349)
python train.py --epochs 50 --dataset UniGakGakDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model vgg11 true --optimizer SGD --name UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_SGD --label_type age
python train.py --epochs 50 --dataset UniGakGakDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model vgg11 true --optimizer SGD --name UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_SGD --label_type gender
python train.py --epochs 50 --dataset UniGakGakDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model vgg11 true --optimizer SGD --name UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_SGD --label_type mask

python inference.py --model PretrainedModel --is_seperated true --name UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_SGD --pre_trained_model vgg11 true


# output_UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_f1
python train.py --epochs 50 --dataset UniGakGakDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model vgg11 true --optimizer SGD --name UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_f1 --label_type age
python train.py --epochs 50 --dataset UniGakGakDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model vgg11 true --optimizer SGD --name UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_f1 --label_type gender
python train.py --epochs 50 --dataset UniGakGakDataset --augmentation BaseAugmentation --model PretrainedModel --pre_trained_model vgg11 true --optimizer SGD --name UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_f1 --label_type mask

python inference.py --model PretrainedModel --is_seperated true --name UniGakGakDataset_BaseAugmentation_PretrainedModel-vgg11-_f1 --pre_trained_model vgg11 true


# 던저보는 거임.
# output_UniformDataset_BaseAugmentation_PretrainedModel-vgg11-_SGD 
python inference.py --model PretrainedModel --pre_trained_model vgg11 true --is_seperated true --name UniformDataset_BaseAugmentation_PretrainedModel-vgg11-_SGD 
# output_UniformDataset_BaseAugmentation_PretrainedModel-squeezenet1_0-_SGD 
python inference.py --model PretrainedModel --pre_trained_model squeezenet1_0 true --is_seperated true --name UniformDataset_BaseAugmentation_PretrainedModel-squeezenet1_0-_SGD 
# output_UniformDataset_BaseAugmentation_PretrainedModel-alexnet-_SGD 
python inference.py --model PretrainedModel --pre_trained_model alexnet true --is_seperated true --name UniformDataset_BaseAugmentation_PretrainedModel-alexnet-_SGD
