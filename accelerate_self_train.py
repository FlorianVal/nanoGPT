from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from src.branchymodel import BranchyLlama

import datasets
import transformers
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from torch.optim import AdamW

import torch 

class RandomIntDataset(Dataset):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return {"input_ids": torch.randint(0, self.vocab_size, (1,))}

def create_dataloader(vocab_size, batch_size=8):
    return DataLoader(RandomIntDataset(vocab_size), batch_size=batch_size)

def training_loop(model):
    
    accelerator = Accelerator()
    
    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        
    dataloader = create_dataloader(hyperparameters["vocab_size"], hyperparameters["batch_size"])
    
    set_seed(hyperparameters["seed"])
    
    optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])
    
    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    num_epochs = hyperparameters["num_epochs"]

    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=hyperparameters["steps_per_epoch"] * num_epochs,
    )
    progress_bar = tqdm(range(num_epochs * hyperparameters["steps_per_epoch"]), disable=not accelerator.is_main_process)

    for epoch in range(num_epochs):
        model.train()
        model.lm_head.requires_grad_(False)
        model.model.requires_grad_(False)
        model.auxiliary_outputs.requires_grad_(True)
        batch = next(iter(dataloader))
        for step in range(hyperparameters["steps_per_epoch"]):
            outputs = model(batch)
            loss = outputs.loss
            lm_head_logits = outputs.logits[-1]
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            batch = torch.cat((batch, torch.multinomial(torch.softmax(lm_head_logits[:, -1, :], dim=-1), 1)), dim=-1)
            
        model.eval()
        batch = next(iter(dataloader))
        eval_loss = 0
        for step in range(hyperparameters["validation_steps"]):
            outputs = model(batch)
            eval_loss += outputs.loss
        loss = eval_loss / hyperparameters["validation_steps"]
        
        accelerator.print(f"Epoch {epoch} loss: {loss.item()}")
            
hyperparameters = {
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "steps_per_epoch": 100,
    "validation_steps": 50,
    "batch_size": 1, # Actual batch size will this x 8
    "seed": 42,
    "vocab_size": 32000,
}

if __name__ == "__main__":
    branchyllamaconf = BranchyLlama.config_class.from_pretrained(
        "openlm-research/open_llama_3b_v2"
    )
    branchyllamaconf.self_supervision = True

    model = BranchyLlama.from_pretrained(
        "openlm-research/open_llama_3b_v2", config=branchyllamaconf
    )
    
    training_loop(model)