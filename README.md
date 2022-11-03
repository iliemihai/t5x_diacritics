# Fine-tuning Romanian T5x model on TPU

In this repo we will learn how to fine-tune a [T5x](https://github.com/google-research/t5x) model on TPUs for Romanian diacritics restauration.

The first section of the readme presents our fine-tuned model, the second sections discusses the challenges we faced when trying to train the model and the third section discusses the training script with PyTorch-XLA. 

## Section 1 - Models and results

We have 1 modelsfine-tuned so far, a [mt5x-base model](https://huggingface.co/iliemihai/mt5-base-romanian-diacritics) . 

[(eval score soon)]()

## Section 2 - Challenges

The biggest challenge when fine-tunning a LLM is the dataset size. We can use Pytorch XLA to create a training process on each core of the TPU. If the data fits in the memory of one TPU core than is easy. You just read the whole dataset in the TPU memory, and the dataloader will parallelize the batches between the TPU cores. Reading the whole dataset into memory will help with random sampling. The dataset will be like a map which links each data point to a certain index. This way we can use a distributed random sampler which will fetch batches from different locations in the dataset and send them to each core of the TPU.

But what if the dataset does not fit into memory. Here we can use datasets streaming and IterableDatasets, which will not have acces to random indexes in the dataset and it will not work by default with distributed sampler. Streaming datasets means that the data is downloaded progressively as you iterate over the dataset. So we had to implement the distributed sampler manualy.

Because Pytorch loads into memory the model's state, the optimizer state and the dataset, in order to be able to train a large model, we had to reduce the optimzer state. That's why we chose the Adafactor optimzer instead of Adam. 

## Section 3 - Example
Let's take an example and say we want to train a T5x base model from scratch.
The setup assumes that you have:
1. a corpus in the HuggingFace dataset format already uploaded and available
2. the corpus is to big to fit in the TPU's memory

 In this case we should use `dataset streaming`. The data is downloaded progressively as you iterate over the dataset. We will use the [diacritics corpus](https://huggingface.co/datasets/dumitrescustefan/diacritic) from Huggingface.

1. Define the  ``Dataset``:

```python
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
```

* ``world_size`` represents the number of TPU cores, in our case 8.

2. Define the string processing function:
   
```python
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
```
* the function will tokenize the string and replace the pad tokens with -100.


3. Define dataset, model, tokenizer, optimzer: 

```python
NUM_EPOCHS = 10
max_length = 256
batch_size = 8
num_workers = 64
save_steps = 50000
model_path = "dumitrescustefan/mt5-base-romanian"
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)
model.config.max_length = max_length
device = xm.xla_device()
model.to(device)
# distributed params

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
```
* use Adafactor optimizer for better results

4. Define traininig loop:

```python
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
```

* In the training loop we will use a distributed sampler which will fetch batches from the straming datasets and will send them to each TPU core.
* We will use the XLA spawn function to launch a separate process on each TPU core



#### TPU setup

After ssh-ing on the TPU machine run directly: 

```bash
cd /usr/share/
sudo git clone -b release/1.10 --recursive https://github.com/pytorch/pytorch 
cd pytorch/
sudo git clone -b r1.10 --recursive https://github.com/pytorch/xla.git
cd xla/
yes | sudo pip3 uninstall torch_xla
yes | sudo pip3 uninstall torch
yes | sudo pip3 uninstall torch_vision
sudo pip3 install torch==1.10.0
sudo pip3 install torchvision==0.11.1
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.10-cp38-cp38-linux_x86_64.whl
sudo mv /usr/lib/libtpu.so /tmp
sudo /snap/bin/gsutil cp gs://tpu-pytorch/v4_wheel/110/libtpu.so /lib/libtpu.so
```

Also don't forget to configure the devices:

```bash
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```

## Acknowledgements

Many thanks to the **Tensorflow Research Credits (TRC) team** without which these models would not have been possible to train and opensource. Their support team was quick and helpful throughout the months I've had TRC credits. If only the TPU on-line documentation was as good as their support :)


_Yours truly,_ 

_[Stefan Dumitrescu](https://github.com/dumitrescustefan), [Mihai Ilie](https://github.com/iliemihai)_

