import random
import time
import torch
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.branchymodel import BranchyLlama
from src.RandomIntDataset import RandomIntDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 8
total_step = 100

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "info",
    "report_to": "none",
}
training_args = TrainingArguments(per_device_train_batch_size=4, fp16=True, sharded_ddp="simple", **default_args)


branchyllamaconf = BranchyLlama.config_class.from_pretrained(
    "openlm-research/open_llama_3b_v2"
)
branchyllamaconf.self_supervision = True

print(f"Loading model with config: {branchyllamaconf}")
model = BranchyLlama.from_pretrained(
    "openlm-research/open_llama_3b_v2", config=branchyllamaconf, device_map=device
)


dataloader = DataLoader(RandomIntDataset(batch_size), batch_size=batch_size, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

accelerator = Accelerator()
model = accelerator.prepare(model)
optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

model.train()
model.lm_head.requires_grad_(False)
model.model.requires_grad_(False)
model.auxiliary_outputs.requires_grad_(True)

batch = next(iter(dataloader))
batch = batch.to(device)
for step in range(total_step):
    outputs = model(batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    lm_head_logits = outputs.logits[-1]
    batch = torch.cat(batch, torch.multinomial(lm_head_logits[:, -1, :], 1))
    print(f"Step {step} loss: {loss.item()}")