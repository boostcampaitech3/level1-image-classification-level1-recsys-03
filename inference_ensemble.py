import argparse
import os
from importlib import import_module
import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import model.model as model_model
from dataset import TestDataset, MaskBaseDataset
from model.loss import create_criterion


def load_model(model_name, saved_model, num_classes, model_param, device):
    model_cls = getattr(model_model, model_name)
    model = model_cls(
        num_classes=num_classes,
        **model_param
    )
    
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    return model


def parse_config(model_dirs):
    # parse config files for model directory
    import json
    from collections import defaultdict
    
    label_numclasses = {
        'age': 3,
        'gender': 2,
        'mask': 3
    }
    
    configs = dict((label, []) for label in label_numclasses.keys())
    for idx, model_dir in enumerate(model_dirs):
        with open(os.path.join(model_dir, 'config.json'), 'r') as jsonfile:
            config = json.load(jsonfile)
            label = config['label']
            config['num_classes'] = label_numclasses[label]
            config['model_dir'] = model_dir
            config['model_name'] = config['model']
            del config['model']
            configs[label].append(config)
    return configs


def set_models(configs):
    # takes as input a list of configurations with same labels, add model within config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_param_module = getattr(import_module("train"), 'parse_model_param')
    for config in configs:
        pretrained = config['model'] in ['VGGFace', 'PretrainedModels']
        model_param = model_param_module(config['model_param'], pretrained)
        model = load_model(
            config['model'],
            config['model_dir'],
            config['num_classes'],
            model_param,
            device
        )
        model = torch.nn.DataParallel(model)
        config['model'] = model


@torch.no_grad()
def inference(data_dir, model_dirs, output_dir, args):
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    configs_label = parse_config(model_dirs)
    for _, config in configs_label.items():
        set_models(config)
    
    assert all([configs_label['age'], configs_label['gender'], configs_label['mask']]) # must have em all
    age_soft, age_hard = inference_model(configs_label['age'], loader)
    gen_soft, gen_hard = inference_model(configs_label['gender'], loader)
    msk_soft, msk_hard = inference_model(configs_label['mask'], loader)
    
    pred_soft = msk_soft*6 + gen_soft * 3 + age_soft
    pred_hard = msk_hard*6 + gen_hard * 3 + age_hard
    
    info['ans'] = pred_soft
    info.to_csv(os.path.join(output_dir, 'soft', args.output_filename), index=False)
    info['ans'] = pred_hard
    info.to_csv(os.path.join(output_dir, 'hard', args.output_filename), index=False)
    print(f'Inference Done!')

import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import model.model as model_model
from dataset import TestDataset, MaskBaseDataset
from model.loss import create_criterion


def load_model(model_name, saved_model, num_classes, model_param, device):
    model_cls = getattr(model_model, model_name)
    model = model_cls(
        num_classes=num_classes,
        **model_param
    )
    
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    return model


def parse_config(model_dirs):
    # parse config files for model directory
    import json
    from collections import defaultdict
    
    label_numclasses = {
        'age': 3,
        'gender': 2,
        'mask': 3
    }
    
    configs = dict((label, []) for label in label_numclasses.keys())
    for idx, model_dir in enumerate(model_dirs):
        with open(os.path.join(model_dir, 'config.json'), 'r') as jsonfile:
            config = json.load(jsonfile)
            label = config['label']
            config['num_classes'] = label_numclasses[label]
            config['model_dir'] = model_dir
            config['model_name'] = config['model']
            del config['model']
            configs[label].append(config)
    return configs


def set_models(configs):
    # takes as input a list of configurations with same labels, add model within config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_param_module = getattr(import_module("train"), 'parse_model_param')
    for config in configs:
        pretrained = config['model_name'] in ['VGGFace', 'PretrainedModels']
        model_param = model_param_module(config['model_param'], pretrained)
        model = load_model(
            config['model_name'],
            config['model_dir'],
            config['num_classes'],
            model_param,
            device
        )
        model = torch.nn.DataParallel(model)
        config['model'] = model


@torch.no_grad()
def inference(data_dir, model_dirs, output_dir, args):
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    configs_label = parse_config(model_dirs)
    for _, config in configs_label.items():
        set_models(config)
    
    assert all([configs_label['age'], configs_label['gender'], configs_label['mask']]) # must have em all
    age_soft, age_hard = inference_model(configs_label['age'], loader)
    gen_soft, gen_hard = inference_model(configs_label['gender'], loader)
    msk_soft, msk_hard = inference_model(configs_label['mask'], loader)
    
    pred_soft = msk_soft*6 + gen_soft * 3 + age_soft
    pred_hard = msk_hard*6 + gen_hard * 3 + age_hard
    
    info['ans'] = pred_soft
    info.to_csv(os.path.join(output_dir, 'soft', args.output_filename), index=False)
    info['ans'] = pred_hard
    info.to_csv(os.path.join(output_dir, 'hard', args.output_filename), index=False)
    print(f'Inference Done!')


def inference_model(configs, loader):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    lst_soft = []
    df_hard = pd.DataFrame()
    for config in configs:
        model = config['model']
        model_name = config['model_dir']
        model.eval()
        
        preds = []
        print(f"Calculating inference results for {model_name}..")
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                pred = pred.cpu().numpy()
                preds.extend(pred)
                pred_hard = pred.argmax(axis=-1)
                df_hard[model_name] = pred_hard
        lst_soft.append(preds)
    np_soft = np.array(lst_soft)
    np_soft = np_soft.sum(axis=1)/np_soft.shape[1]
    np_soft = np_soft.argmax(axis=-1)
    np_hard = np.asarray(df_hard.mode(axis=1)[0])

    return np_soft, np_hard


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=500, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--output_filename', type=str, default='output.csv')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dirs', nargs='+', default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dirs = args.model_dirs
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dirs, output_dir, args)
