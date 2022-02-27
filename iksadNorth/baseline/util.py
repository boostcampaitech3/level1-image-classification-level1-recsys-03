# %%
import logging
import time
import wandb

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# %%
def see(func):
    
    def wrapper(*args, **kwargs):
        start = time.time()
        
        result = func(*args, **kwargs)

        logger.debug("=" * 50)
        logger.debug(f"해당 함수 이름   :   {func.__qualname__}")
        logger.debug(f"작동 시간        :   {time.time() - start}")
        return result
    
    return wrapper

# %%
from pathlib import Path
import random

class folder():
    SUB_CHAR = "."
    _file_names = {
        "mask1": 0,
        "mask2": 0,
        "mask3": 0,
        "mask4": 0,
        "mask5": 0,
        "incorrect_mask": 1,
        "normal": 2
    }
    
    def __init__ (self, path):
        if isinstance(path, Path):
            self.path = path
        elif isinstance(path, str):
            self.path = Path(path)
        self.name = self.path.stem
        self.child = {}
        
        self.mask = []
        self.normal = []
        self.incorrect_mask = []
        
        self._setup()
        self._merge()
    
    def _substract(self, iter_dir, trmf=list):
        """Path.iterdir()을 인수로 받아서 SUB_CHAR으로 시작하는 것 파일들 제외.

        Args:
            iter_dir (Path.iterdir()): Path.iterdir()을 위한 함수임.
            trmf (_type_, optional): _description_. Defaults to list.

        Returns:
            _type_: _description_
        """
        res = filter(lambda x: not str(x.stem).startswith(self.SUB_CHAR), iter_dir)
        return trmf(res)
    
    def _setup(self):
        """해당 경로의 하위 파일 객체 쌓기.
        """
        path = self.path.iterdir()
        ls_img = self._substract(path)
        
        for item in ls_img:
            val = self._file_names[item.stem]
            if val==0:
                self.mask.append(item)
            elif val==1:
                self.incorrect_mask.append(item)
            elif val==2:
                self.normal.append(item)
                
    def _merge(self):
        """images의 모든 img 쌓기
        """
        self.child['mask'] = (self.mask)
        self.child['normal'] = (self.normal)
        self.child['incorrect_mask'] = (self.incorrect_mask)
        
    def list(self):
        result = []
        for i in self.child.values():
            result.extend(i)
        return result
    
    def label(self):
        id, gender, race, age = self.name.split('_')
        return gender, int(age)

    def pick(self):
        result = {}
        one = random.choices(self.mask, k = 1)[0]
        
        result[one.stem] = one
        result["incorrect_mask"] = self.incorrect_mask[0]
        result["normal"] = self.normal[0]
        
        return result

# %%
class images_folder():
    SUB_CHAR = "."
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.stem
        
        self.child = []
        
        self._setup()
            
    def _substract(self, iter_dir, trmf=list):
        """Path.iterdir()을 인수로 받아서 SUB_CHAR으로 시작하는 것 파일들 제외.

        Args:
            iter_dir (Path.iterdir()): Path.iterdir()을 위한 함수임.
            trmf (_type_, optional): _description_. Defaults to list.

        Returns:
            _type_: _description_
        """
        res = filter(lambda x: not str(x.stem).startswith(self.SUB_CHAR), iter_dir)
        return trmf(res)
    
    def _setup(self):
        """해당 경로의 하위 파일 객체 쌓기.
        """
        path = self.path.iterdir()
        ls_img = self._substract(path)
        
        for item in ls_img:
            self.child.append(item)    
    
    def get(self, index):
        return folder(self.child[index])

# %%
import torch
from collections import deque

class EarlyStop():
    def __init__(self, atol:float=0, rtol:float=1e-08, length:int=10):
        self.atol = atol
        self.rtol = rtol
        self.length = length
        
        self.queue = deque([float('inf')])
    
    def append(self, val:float):
        assert isinstance(val, (int, float)), "Earlystop.append()에는 1개의 값만 들어와야 한다."
        assert not torch.isnan(torch.tensor(val)).all(), "Earlystop.append()에 Nan 값이 들어옴."
        
        self.queue.append(val)
        while self.queue.__len__() > self.length:
            self.queue.popleft()
        return self.isclose()
    
    def isclose(self):
        max_ = max(self.queue)
        min_ = min(self.queue)
        
        max_, min_ = torch.Tensor([max_]), torch.Tensor([min_])
        return torch.isclose(max_, min_, rtol=self.rtol, atol=self.atol).item()
        
        

# %%
import torch
import torch.nn as nn
from pprint import pprint

from typing import Callable
from typing import Optional
from typing import Union

from functools import partial

import wandb

# %%
class Hook(object):
    """pytorch의 hook을 좀더 쉽게 사용하고자 제작한 클래스.
    hook들을 한 곳에 모아 .remove()하기 쉽게 설계했으며(class FishingKit 참고),
    hook 객체가 del 명령어로 삭제될 때 자동으로 hook_fn를 가진 Tensor or module에서 제거되도록 설계함.
    
    뿐만 아니라 해당 nn.Moduel과 torch.Tensor에 쉽게 .register_XXX_hook()을 하기 위해 
    별도의 메서드를 추가함. 다음이 그러한 메서드들이다.
    
    - .insert(module)
    - .attach(module)
    - .tag(tensor)
    """
    def __init__(self, kit, name:str=None, fn:Callable[[nn.Module, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]=None):
        """객체 생성 시, 해당 Kit에 등록하고 name 필드와 fn 필드를 초기화한다.
        이 때, name 값이 None이라면, 임의의 10자리 난수를 이름으로 지정한다.
        fn 필드는 hook_fn으로 사용하고 싶은 함수를 의미하는데 만약 해당 값이 주어지지 않는다면 기존에 정의된 hook_fn을 그대로 사용한다.

        Args:
            kit (FishingKit): 해당 Hook 객체가 관리될 Kit 객체
            name (str, optional): 해당 Hook 객체의 이름. Defaults to None.
            fn (Callable[[nn.Module, torch.Tensor, torch.Tensor], Optional[torch.Tensor]], optional): hook_fn으로 사용하고 싶은 함수. Defaults to None.
        """
        self.name = name if name else str(int(random.random() * 1e10))
        self.hook = None
        self.kit = kit
        self.kit.append(self)
        self.hook_fn = fn if fn else self.hook_fn
        
    def hook_fn(self, module, input, output):
        wandb.log({
            'module_time:':time.time()
        })
    
    # =================================================================================
    # 아래는 hook 객체가 del 명령어로 삭제될 때 자동으로 hook_fn를 가진 Tensor or module에서 제거되도록 설계한 메서드들.
    
    def close(self):
        """해당 객체의 hook객체가 None이 아니라면 삭제한다.
        """
        if self.hook:
            self.hook.remove()
        
    def __del__(self):
        self.close()
    
    # =================================================================================
    # 아래는 쉽게 .register_XXX_hook()을 쉽게 사용하기 위해 별도로 추가한 메서드들.
    
    def insert(self, module:nn.Module, forward:bool=True):
        """.register_forward_hook()을 쉽게 사용하기 위한 메서드.
        insert가 억지로 사이에 집어넣는다는 뉘앙스가 있어 forward에 사용하게 됨.
        
        forward가 False일 때는 .register_full_backward_hook()를 사용하게 된다.

        Args:
            module (nn.Module): forward_hook or backward_hook을 달아줄 Module.
            forward (bool, optional): forward_hook를 사용할지 여부. 아니라면 backward_hook 사용. Defaults to True.
        """
        if forward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            if "register_full_backward_hook" in module.__dir__():
                self.hook = module.register_full_backward_hook(self.hook_fn)
            else:
                self.hook = module.register_backward_hook(self.hook_fn)
                
    
    def attach(self, module:nn.Module):
        """.register_full_backward_hook()을 쉽게 사용하기 위한 메서드.
        attach가 겉에 무심하게 붙인다는 뉘앙스가 있어 backward에 사용하게 됨.

        Args:
            module (nn.Module): backward_hook을 달아줄 Module
        """
        self.insert(module, forward=False)

    def tag(self, tensor:torch.Tensor):
        """.register_hook()을 쉽게 사용하기 위한 메서드.
        tag가 태그를 붙이고 감시한다는 뉘앙스가 있어 tensor에 사용하게 됨.

        Args:
            tensor (torch.Tensor): backward_hook을 달아줄 Tensor
        """
        hook_fn = self.hook_fn
        hook_fn = partial(hook_fn, tensor, 'None because of type, tensor')
        self.hook = tensor.register_hook(hook_fn)

class FishingKit(object):
    """Hook들을 일괄 관리하는 곳. 차후에 일괄 print하거나 save하는 기능을 추가할 예정
    """
    def __init__(self, name:str):
        """FishingKit의 name 필드와 box 필드를 초기화시킴.

        Args:
            name (str): 해당 FishingKit의 이름
        """
        self.name = name
        self._box = {}
    
    @property
    def box(self):
        return self._box
    
    def __getitem__(self, key:Union[int, str]) -> Hook:
        """Kit 내부에 있는 Hook을 List or Dict의 방식으로 인덱싱함.

        Args:
            key (Union[int, str]): 보통 Hook.name을 key값으로 쓰지만 특별히 숫자 인덱스도 사용가능하게 만듦.

        Returns:
            Hook: 인덱싱 결과로 나온 value값.
        """
        if key in self._box.keys():
            return self._box[key]
        else:
            new_dict = {idx:hooks for idx, hooks in enumerate(self._box.values())}
            return new_dict[key]
    
    def append(self, hook:Hook):
        """box 필드에 Hook 객체를 추가시킴. 
        이때, box는 dict 형태인데 
        key 값은 Hook 객체의 name 필드를 사용하고
        value는 Hook 객체이다.

        Args:
            hook (Hook): 해당 kit에 관리보관하고 싶은 Hook 객체
        """
        self.box[hook.name] = hook
        
    def hook(self, name:str=None, fn:Callable[[nn.Module, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]=None) -> Hook:
        """해당 Kit에 종속된 Hook 객체를 생성. 생성된 Hook 객체는 별도로 Kit에 .append()하지 않아도 된다.

        Args:
            name (str, optional): 생성될 Hook의 이름. Defaults to None.
            fn (Callable[[nn.Module, torch.Tensor, torch.Tensor], Optional[torch.Tensor]], optional): 생성될 Hook의 hook_fn. Defaults to None.

        Returns:
            Hook: 해당 Kit에 종속된 Hook 객체.
            
            사용 예시) 
            fk.hook('forward').insert(linear)
            fk.hook('backward').attach(linear)
            fk.hook('tensor').tag(a)
        """
        return Hook(self, name, fn)

# %%
if __name__ == '__main__':
    linear = nn.Linear(3,4,True)
    arg = nn.Parameter(torch.full((10,3), 7.0))
    
    
    kit = FishingKit('print_grad')
    
    kit.hook('forward').insert(linear)
    kit.hook('backward').attach(linear)
    kit.hook('tensor').tag(arg)
    pprint(kit.box)
    pprint('*'*50)
    
    pprint(kit['forward'].name)
    pprint(kit[0].name)
    pprint('*'*50)
    
    m = linear(arg)
    loss = torch.sum(m)
    loss.backward()

# %%
if __name__ == '__main__':
    @see
    def add(a,b):
        return a + b
    add(1,2)

    k = folder('/opt/ml/input/data/train/images/006927_male_Asian_19')
    n = folder(Path('/opt/ml/input/data/train/images/006927_male_Asian_19'))
    m = images_folder('/opt/ml/input/data/train/images')


# %%



