from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple

import torch

@dataclass
class CausalBranchyLLMOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    head_loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    head_outputs: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
   