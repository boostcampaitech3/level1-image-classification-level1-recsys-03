import argparse
import os
from importlib import import_module
from xmlrpc.client import boolean

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device, **kwargs):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes,
        **kwargs
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args, **kwargs):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device, **kwargs).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,#8
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


@torch.no_grad()
def inference_seperated_one(data_dir, model_dir, output_dir, args, **kwargs):
    """
    model_dir 디렉토리 아래에는 [model_dir/age], [model_dir/gender], [model_dir/mask]가 포함되어 있어야 합니다.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ###################################################################################
    num_classes = 3
    model_age = load_model(model_dir + '/age', num_classes, device, **kwargs).to(device)
    model_age.eval()
    
    num_classes = 2
    model_gender = load_model(model_dir + '/gender', num_classes, device, **kwargs).to(device)
    model_gender.eval()
    
    num_classes = 3
    model_mask = load_model(model_dir + '/mask', num_classes, device, **kwargs).to(device)
    model_mask.eval()
    ###################################################################################
    
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,#8
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            
            pred_age = model_age(images).argmax(dim=-1)
            pred_gender = model_gender(images).argmax(dim=-1)
            pred_mask = model_mask(images).argmax(dim=-1)
            
            pred = MaskBaseDataset.encode_multi_class(pred_mask, pred_gender, pred_age)
                  
            pred = pred#.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96*4, 128*4), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='ResNetModel', help='model type (default: BaseModel)')

    # Container environment
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    # parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    # parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--model_dir', type=str, default='/opt/ml/workspace/saved')
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/eval')
    parser.add_argument('--output_dir', type=str, default='/opt/ml/workspace/output')
    
    # 내가 추가한 것.
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--saved_dir', type=str, default=None)
    parser.add_argument('--is_seperated', type=boolean, default=False)
    parser.add_argument('--label_type', type=str, default=None, help='dataset에서 사용하고 싶은 label. "age", "gender", "mask" 중 택1')
    parser.add_argument('--saved_dir_model', type=str, default=None, help='학습시킨 모델을 다시 로드함. 해당 경로는 사용하고 싶은 모델의 .pth를 정확히 기술해야 함.')
    parser.add_argument('--pre_trained_model', nargs='*', default=None, help='torchvision.model에 존재하는 model과 freeze 여부.')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = os.path.join(args.model_dir, args.name)
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    kwargs = {}
    
    if args.saved_dir:
        kwargs['saved_dir'] = args.saved_dir
    if args.label_type:
        kwargs['label_type'] = args.label_type
    if args.saved_dir_model:
        kwargs['saved_dir_model'] = args.saved_dir_model
    if args.pre_trained_model:
        kwargs['model_using'] = args.pre_trained_model[0]
        kwargs['freeze'] = args.pre_trained_model[1]
    
    
    if args.is_seperated:
        inference_seperated_one(data_dir, model_dir, output_dir, args, **kwargs)
    else:
        inference(data_dir, model_dir, output_dir, args, **kwargs)
