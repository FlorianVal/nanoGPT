import random
import time
import torch
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoTokenizer

from src.branchymodel import BranchyLlama
from src.RandomIntDataset import RandomIntDataset
import wandb

batch_size = 8
total_step = 100

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "info",
    "report_to": "none",
}
training_args = TrainingArguments(sharded_ddp="simple", **default_args)

wandb.init(project="branchyLlama", config={"batch_size": batch_size, "total_step": total_step, "training_args": training_args})

branchyllamaconf = BranchyLlama.config_class.from_pretrained(
    "openlm-research/open_llama_3b_v2"
)
branchyllamaconf.self_supervision = True

print(f"Loading model with config: {branchyllamaconf}")
model = BranchyLlama.from_pretrained(
    "openlm-research/open_llama_3b_v2", config=branchyllamaconf
)


dataloader = DataLoader(RandomIntDataset(batch_size), batch_size=batch_size, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_step, eta_min=0)

accelerator = Accelerator()
model = accelerator.prepare(model)
optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")

model.train()
model.lm_head.requires_grad_(False)
model.model.requires_grad_(False)
model.auxiliary_outputs.requires_grad_(True)

batch = next(iter(dataloader))

for step in range(total_step):
    outputs = model(batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    lm_head_logits = outputs.logits[-1]
    loss_per_head = {i: head_loss.item() for i, head_loss in enumerate(outputs.head_loss)}
    wandb.log({"loss": loss.item(), "loss_per_head": loss_per_head})
    batch = torch.cat((batch, torch.multinomial(torch.softmax(lm_head_logits[:, -1, :], dim=-1), 1)), dim=-1)
    print(f"Step {step} loss: {loss.item()} and batch shape : {batch.shape}")
    #print([tokenizer.decode(sentence.squeeze().tolist()) for sentence in batch])
