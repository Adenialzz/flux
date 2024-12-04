import torch
from dataclasses import dataclass, field
from typing import Union, Dict

@dataclass
class RFEditInfo:
    inverse: bool = False
    block_type: str = 'single'
    block_id: int = -1
    inject: bool = False
    inject_step: int = 0
    # feature_path: str = 'feature_path'
    t: float = 0
    second_order = False
    feature: Dict[str, torch.Tensor] =  field(default_factory=dict) 
