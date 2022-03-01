
# weighted focal loss
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion focal --label mask --name mask_focal
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion focal --label gender --name gender_focal
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion focal --label age --name age_focal

# label smoothing
python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label mask --name mask_label_smoothing
python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label gender --name gender_label_smoothing
python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label age --name age_label_smoothing

