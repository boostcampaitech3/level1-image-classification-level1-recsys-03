# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision



# %%

class BaseModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)



# %%

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x



# %%

class ResNetModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self._num_classes = num_classes
        self.resnet18 = torchvision.models.resnet18(pretrained=True, num_classes=1000)
        self._haircut(self.num_classes)
        
    def forward(self, x):
        return self.resnet18(x)
    
    def _haircut(self, num_classes):
        self.resnet18.fc = torch.nn.Linear(in_features = 512, out_features = num_classes, bias = True)
        torch.nn.init.xavier_uniform_(self.resnet18.fc.weight)
        stdv = 1. / (self.resnet18.fc.weight.size(1)) ** 0.5
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)
    
    @property
    def num_classes(self):
        return self._num_classes

# %%
class Vgg19Model(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self._num_classes = num_classes
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        self._haircut(self.num_classes)
        
    def forward(self, x):
        return self.vgg19(x)
    
    def _haircut(self, num_classes):
        self.vgg19.classifier[-1] = torch.nn.Linear(in_features = 4096, out_features = num_classes, bias = True)
        torch.nn.init.xavier_uniform_(self.vgg19.classifier[-1].weight)
        stdv = 1. / (self.vgg19.classifier[-1].weight.size(1)) ** 0.5
        self.vgg19.classifier[-1].bias.data.uniform_(-stdv, stdv)
    
    @property
    def num_classes(self):
        return self._num_classes

# %%
from torchvision import models as repo_pretrained_model
from torch import nn

# Custom Model Template
class PretrainedModel(nn.Module):
    
    # 이 모델의 위한 환경변수들.
    NUM_CLASSES_ASSUMED=1000
    HIDDEN_LAYER_FEATURES=100
    
    def __init__(self, num_classes, model_using:str='resnet18', pretrained:bool=True , freeze:bool=True,**kwargs):
        assert hasattr(repo_pretrained_model, model_using), f"해당 model_using({model_using})은 torchvision.model에 존재하지 않음."
        
        super().__init__()
        # 파라미터들 필드로 선언.
        self.num_classes = num_classes
        self.model_using = model_using
        self.pretrained = pretrained
        
        # 출처
        # https://github.com/pytorch/vision/tree/main/torchvision/models
        # 해당 모델들의 num_classes가 항상 1000이라는 가정하에 설계됨.
        # 위 가정은 꼼꼼히 확인한 것이 아니기 때문에 오류가 날 수 있음.
        self.model = getattr(repo_pretrained_model, model_using)
        print(f"{model_using} :: {self.model.__qualname__}")
        
        self.net_pretrained = self.model(pretrained, **kwargs)
        print(f"net_pretrained :: {self.net_pretrained}")
        if freeze:
            self._freeze()
        
        # 해당 모델들의 num_classes가 항상 1000이라는 가정하에 설계됨. 
        # ONLY (num_classes==1000)
        self.net_mlp = nn.Sequential(
            nn.Linear(self.NUM_CLASSES_ASSUMED, self.HIDDEN_LAYER_FEATURES, bias=True),
            nn.GELU(),
            nn.Linear(self.HIDDEN_LAYER_FEATURES, self.num_classes, bias=True),
        )
        
        # pretrained module과 MLP의 결합.
        self.sequential = nn.Sequential(
            self.net_pretrained,
            self.net_mlp,
        )
        print(f"final_model :: {self.sequential}")


    def forward(self, x):
        x = self.sequential(x)
        return x
    
    
    def _freeze(self):
        net = self.net_pretrained
        
        for params in net.parameters():
            params.require_grad = False
    
    
    def _melt(self):
        net = self.net_pretrained
        
        for params in net.parameters():
            params.require_grad = True
    
    
    def _melt_gradually(self, idx_elapsed:int):
        net = self.net_pretrained
        
        # 고안 예정.
        pass

# %%
from coatnet import CoAtNet
from torchvision.transforms import Resize, CenterCrop, Compose, ToPILImage, ToTensor

class CoAtNetModel(nn.Module):
    IMAGE_SIZE = (512, 384)
    AFTER_TRANS = (224, 224)
    
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self._num_classes = num_classes
        
        num_blocks = [2, 2, 3, 5, 2]    
        channels = [64, 96, 192, 384, 768] 
        self.coatnet = CoAtNet(self.AFTER_TRANS, 3, num_blocks, channels, num_classes=self._num_classes)
        
        self.trfm = Compose([
            ToPILImage(),
            Resize(min(self.AFTER_TRANS)),
            CenterCrop(size=self.AFTER_TRANS),
            ToTensor()
        ])
        
    def forward(self, x):
        device = x.device
        x = x.cpu()
        x_ = [self.trfm(img) for img in x]
        x = torch.stack(x_, dim=0)
        x = x.to(device)
        return self.coatnet(x)
    
    @property
    def num_classes(self):
        return self._num_classes

# %%
from dataset import MaskBaseDataset

class MergeLabel(nn.Module):
    def __init__(self, num_classes, saved_dir, **kwargs):
        super(MergeLabel, self).__init__()
        self.age_model = ResNetModel(3)
        self.gender_model = ResNetModel(2)
        self.mask_model = ResNetModel(3)
        
        age_dir=f'{saved_dir}/age/best.pth'
        gender_dir=f'{saved_dir}/gender/best.pth'
        mask_dir=f'{saved_dir}/mask/best.pth'
        
        self.age_model.load_state_dict(torch.load(age_dir))
        self.gender_model.load_state_dict(torch.load(gender_dir))
        self.mask_model.load_state_dict(torch.load(mask_dir))

    @torch.no_grad()
    def forward(self, x):
        N = x.shape[0]
        dtype = x.dtype
        device = x.device
        
        age = self.age_model(x)
        gender = self.gender_model(x)
        mask = self.mask_model(x)
        
        for label in [age, gender, mask]:
            label.to(device)
        
        # x = [MaskBaseDataset.encode_multi_class(mask[i], gender[i], age[i]) for i in range(x.shape[0])]
        x = torch.zeros(N)
        x = x.to(device).type(dtype)
        for i in range(N):
            one_hot_encoding = mask[i].argmax(), gender[i].argmax(), age[i].argmax()
            x[i] = MaskBaseDataset.encode_multi_class(*one_hot_encoding)
        
        x = F.one_hot(x.long(), num_classes=18)
        return x.float()#.to(device)

# %%
from dataset import MaskBaseDataset
from dataset import CustomAugmentation

if __name__ == '__main__':
    base = BaseModel(18)
    
    age = ResNetModel(3)
    gender = ResNetModel(2)
    mask = ResNetModel(3)
    
    # total = MergeLabel(None, "/opt/ml/workspace/baseline/model")
    
    vgg = Vgg19Model(18)
    coatnet = CoAtNetModel(18)
    
    pretrained_model = PretrainedModel(3)

    dataset = MaskBaseDataset('/opt/ml/input/data/train/images')
    dataset.set_transform(CustomAugmentation((384, 512), mean=1.0, std=1.0))

    x = dataset[0][0].unsqueeze(0)
    

# %%



