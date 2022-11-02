import torch
import math
from torch.utils.data._utils.worker import get_worker_info
from torch.utils.data import Dataset, IterableDataset
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import (T5TokenizerFast, AutoTokenizer, MT5ForConditionalGeneration)
from datasets import load_dataset
import torch_xla.distributed.parallel_loader as pl
import torch
import unidecode
import random
import pandas as pd
from transformers.optimization import Adafactor
from torch.utils.data.distributed import Sampler
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader 
from datasets import load_dataset
from tqdm import tqdm


class DistributedIterableDataset(IterableDataset):
    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker_info = get_worker_info()
        mod = self.world_size
        shift = self.rank

        if worker_info:
            mod *= worker_info.num_workers
            shift = self.rank * worker_info.num_workers + worker_info.id

        ind = 0
        for data in self.dataset:
            ind += 1
            if (ind + shift) % mod == 0:
                yield {
                        "source": unidecode.unidecode(data["text"])+"</s>",
                        "target": data["text"]+"</s>"
                      }
                
NUM_EPOCHS = 10
max_length = 256
batch_size = 8
num_workers = 64
save_steps = 50000
model_path = "dumitrescustefan/mt5-base-romanian"
tokenizer = T5TokenizerFast.from_pretrained(model_path)

def my_collate(batch):
    text_batch_source = []
    text_batch_target = []
    for instance in batch:

        text_batch_source.append(instance["source"])
        text_batch_target.append(instance["target"])

    text_batch_source_out = tokenizer(text_batch_source,
                           max_length=max_length, truncation=True, padding="max_length", add_special_tokens=True, return_tensors="pt")
    text_batch_target_out = tokenizer(text_batch_target,
                                      max_length=max_length, truncation=True, padding="max_length", add_special_tokens=True,return_tensors="pt")

    text_batch_source_out["input_ids"][text_batch_source_out["input_ids"][:, :] == tokenizer.pad_token_id] = -100
    text_batch_source_out["input_ids"][text_batch_source_out["input_ids"][:, :] == tokenizer.pad_token_id] = -100

    return text_batch_source_out, text_batch_target_out


def main():
    # model
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    model.config.max_length = max_length
    device = xm.xla_device()
    model.to(device)
    print('Device: ', device)
    # distributed params
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    torch.distributed.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:5678',
        world_size=world_size,
        rank=rank,
    )
    # data
    train_data = load_dataset("dumitrescustefan/diacritic", split="train", streaming=True)

    train_dataset = DistributedIterableDataset(train_data, rank, world_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=my_collate, pin_memory=True, drop_last = True)

    optimizer = Adafactor(
                    model.parameters(),
                    lr=1e-4,
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False,
               )
    step = 0
    # Training
    for epoch in tqdm(range(NUM_EPOCHS)):
        xm.master_print(f"Epoch:", epoch)
        para_loader = pl.ParallelLoader(train_dataloader, [device])

        for batch in tqdm(para_loader.per_device_loader(device)):
            model.train()

            batch_source, batch_target = batch[0], batch[1]
            lm_labels = batch_target["input_ids"].to(device)

            input_ids = batch_source["input_ids"].to(device)
            attention_mask_enc = batch_source["attention_mask"].to(device)
            labels = batch_target["input_ids"].to(device)
            attention_mask_dec = batch_target['attention_mask'].to(device)
            optimizer.zero_grad()

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask_enc,
                            labels=labels, 
                            decoder_attention_mask=attention_mask_dec)
            loss = outputs.loss
            xm.master_print("Loss:", loss.item())
            loss.backward()
            optimizer.step()
            xm.mark_step()
            step += 1

            if step % save_steps == 0:
                model.save_pretrained("finetuned_t5_diacritics_"+str(save_steps))



# Start training processes
def _mp_fn(rank, flags):
    main()

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
