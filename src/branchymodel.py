from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaModel, LlamaPreTrainedModel

from .BranchyLlamaConfig import BranchyLlamaConfig
from .LLMDataclass import CausalBranchyLLMOutputWithPast


class BranchyLlama(LlamaPreTrainedModel):
    """
    Llama model with additional auxiliary heads.

    Args:
    ----
        config : LlamaConfig
    """

    config_class = BranchyLlamaConfig

    def __init__(self, config: BranchyLlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)

        self.auxiliary_outputs = nn.ModuleList(
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_hidden_layers)
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        loss_fct: nn.Module,
    ) -> torch.FloatTensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if (
            len(logits.shape) == 4
        ):  # if the logits have an additional dimension for the heads
            # repeat the labels to match the shape of logits
            shift_labels = shift_labels.unsqueeze(0).repeat(logits.shape[0], 1, 1)
        else:
            # normal, single head setting
            shift_labels = shift_labels.unsqueeze(0)
            shift_logits = shift_logits.unsqueeze(0)
        head_number = shift_logits.shape[0]

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return loss.view(head_number, -1).mean(dim=-1)

    def compute_self_supervision_loss(
        self,
        aux_logits: torch.Tensor,
        lm_logits: torch.Tensor,
        return_per_head: bool = False,
    ) -> Dict[str, torch.Tensor]:
        last_aux_logits = aux_logits[..., -1, :]
        last_lm_logits = lm_logits[..., -1, :]

        last_lm_logits = last_lm_logits.repeat(last_aux_logits.shape[0], 1, 1, 1)
        losses = []
        if return_per_head:
            for head_logit in last_aux_logits:
                losses.append(
                    nn.KLDivLoss(reduction="batchmean")(
                        F.log_softmax(head_logit, dim=-1),
                        F.softmax(last_lm_logits[0], dim=-1),
                    )
                )
            loss = torch.stack(losses, dim=0).mean(dim=-1)
        else:
            # Compute the KL divergence between the last auxiliary head and the last LM head
            loss = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(last_aux_logits.view(-1, self.config.vocab_size), dim=-1),
                F.softmax(last_lm_logits.view(-1, self.config.vocab_size), dim=-1),
            )

        return {"loss": loss, "aux_loss": torch.stack(losses)}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        self_supervision: Optional[bool] = True,
    ) -> Union[Tuple, CausalBranchyLLMOutputWithPast]:
        print(self_supervision)
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if not output_hidden_states:
            raise ValueError("output_hidden_states must be True for BranchyLlama")
        if self_supervision and labels is not None:
            raise ValueError(
                "self_supervision and labels cannot be specified at the same time"
            )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_states = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        """ if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            auxiliary_head_slices = [
                head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                for head in self.auxiliary_outputs
            ]
            # Slice every head
            aux_logits = [
                [
                    F.linear(hidden_states[i], auxiliary_head_slices[i][j])
                    for j in range(self.config.pretraining_tp)
                ]
                for i in range(self.config.num_hidden_layers)
            ]
            aux_logits = [
                torch.cat(aux_logits[i], dim=-1)
                for i in range(self.config.num_hidden_layers)
            ]

            lm_logits = [
                F.linear(last_hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            lm_logits = torch.cat(lm_logits, dim=-1)

        else: """
        aux_logits = [
            head(hidden_states[i]) for i, head in enumerate(self.auxiliary_outputs)
        ]
        lm_logits = self.lm_head(last_hidden_states)

        aux_logits = torch.stack(aux_logits, dim=0).float()
        lm_logits = lm_logits.float()
        logits = torch.cat([aux_logits, lm_logits.unsqueeze(0)], dim=0)

        loss = None
        lm_loss = None
        aux_loss = None
        if labels is not None:
            # Compute loss as in Llama implementation
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = self.compute_loss(lm_logits, labels, loss_fct)
            aux_loss = self.compute_loss(aux_logits, labels, loss_fct)
            loss = torch.stack([aux_loss, lm_loss], dim=0)
        if self_supervision:
            # Compute the loss to train the auxiliary heads
            loss = self.compute_self_supervision_loss(aux_logits, lm_logits)
            loss = loss["loss"]
            aux_loss = loss["aux_loss"]
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalBranchyLLMOutputWithPast(
            loss=loss,
            lm_loss=lm_loss,
            head_loss=aux_loss,
            logits=logits,
            head_outputs=aux_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
