##################################### baseline code #####################################

from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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

###########################################################################################
# pretrained VGGFace (base VGG16)
# https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth
###########################################################################################
class VGGFace(nn.Module):
    def __init__(self, num_classes, dict_weight: OrderedDict=None):
        super().__init__()
        self.model = models.vgg16()
        self.num_classes = num_classes
        self.dict_weight = dict_weight
        
        self.init_weights()
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        return self.model(x)
    
    def print(self):
        import numpy as np
        
        np.set_printoptions(precision=3)
        n_param = 0
        for p_idx,(param_name,param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                param_numpy = param.detach().cpu().numpy() # to numpy array 
                n_param += len(param_numpy.reshape(-1))
                print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
                print ("    val:%s"%(param_numpy.reshape(-1)[:5]))
        print ("Total number of parameters:[%s]."%(format(n_param,',d')))
    
    def init_weights(self):
        from torch.utils import model_zoo
        
        # load weights and update label for the loaded state dict (weights)
        if self.dict_weight is None:
            weight_url = 'https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth'
            self.dict_weight = model_zoo.load_url(weight_url)
        vgg_labels = lst = [name for name, _ in self.model.named_parameters() if name.split(sep='.')[0]=='features']
        vggface_weights = list(self.dict_weight.items())
        vggface_weights = [(vgg_labels[idx], vggface_weights[idx][1]) for idx in range(len(vgg_labels))]
        
        self.model.load_state_dict(dict(vggface_weights), strict=False) # strict=False.. otherwise it raises Key Error
        self.model.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes)
        )
        
        
# ResNet18 (pretrained)
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        import math
        
        super().__init__()
        self.num_classes = num_classes
        
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=self.num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)


# # Custom Model Template
# class MyModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         """
#         1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
#         2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
#         3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
#         """

#     def forward(self, x):
#         """
#         1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
#         2. 결과로 나온 output 을 return 해주세요
#         """
#         return x


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
