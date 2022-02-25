import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from dataset import MaskBaseDataset


class BaseModel(nn.Module):
    def __init__(self, num_classes):
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


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
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


class ResNetModel(nn.Module):
    def __init__(self, num_classes):
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


class MergeLabel(nn.Module):
    def __init__(self, num_classes, saved_dir):
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
        
        x = torch.zeros(N)
        x = x.to(device).type(dtype)
        for i in range(N):
            one_hot_encoding = mask[i].argmax(), gender[i].argmax(), age[i].argmax()
            x[i] = MaskBaseDataset.encode_multi_class(*one_hot_encoding)
        
        x = F.one_hot(x.long(), num_classes=18)
        return x.float()
