from transformers import AutoModelForCausalLM, TrainingArguments, get_cosine_schedule_with_warmup
from torchmetrics.functional.classification import accuracy
from trl import DPOTrainer, create_reference_model
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from typing import List
import lightning as pl
import pickle
import random
import torch
import wandb
import time
import os


def do_train(
        dpo_ds,
        tokenizer,
        model_name_or_path,
        batch_size = 2,
        gradient_accumulation_steps = 16,
        max_epochs = 10,
        lr = 3e-5,
        deterministic = True
    ):
    load_dotenv()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        token=os.getenv('hf_personal_access_token')
    )
    base_model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base_model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "right"
    print(f"tokenizer.pad_token: {tokenizer.pad_token}")
    print(f"tokenizer.eos_token: {tokenizer.eos_token}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    adapter_name = "injected_adapter"
    model = get_peft_model(base_model, peft_config, adapter_name)
    model_ref = create_reference_model(base_model)

    training_args = TrainingArguments(
        output_dir="tmp_model",
        overwrite_output_dir=True,
        full_determinism=deterministic,
        do_train=True,
        do_eval=False,
        prediction_loss_only=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        optim='adamw_torch', # 'adamw_bnb_8bit',
        learning_rate=lr,
        weight_decay=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        bf16 = True,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        num_train_epochs=max_epochs,
        logging_strategy='steps',
        logging_steps=0.1,
        logging_first_step=True,
        save_strategy='steps',
        save_steps=0.1,
        save_total_limit=1,
        report_to='wandb',
        disable_tqdm=False,
        push_to_hub=False
    )
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=dpo_ds,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=256
    )
    dpo_trainer.train()

    saved_model_path = "trained_" + str(time.time()).replace('.', '_')
    dpo_trainer.save_model(saved_model_path)
    wandb.finish()

    return saved_model_path

class RouterModel(torch.nn.Module):
    def __init__(self, num_adapters):
        super().__init__()
        self.num_adapters = num_adapters
        self.input_net = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    4096,
                    4,
                    activation=torch.nn.functional.relu,
                    batch_first=True
                ),
                2
            )
        )
        self.output_net = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_adapters)
        )

    def forward(self, features):
        features: torch.Tensor = self.input_net(features)
        if features.dim() == 2:
            features = features.permute(1, 0).mean(dim=-1)
        else:
            features = features.permute(0, 2, 1).mean(dim=-1)
        output: torch.Tensor = self.output_net(features)
        return output

class RouterModule(pl.LightningModule):
    def __init__(
            self, router_model, lr: float = 3e-4, max_epochs: int = 1, train_dataloader_len: int = 100, warmup_proportion: float = 0.1
        ):
        super().__init__()
        self.router_model = router_model
        self.lr = lr
        self.max_epochs = max_epochs
        self.train_dataloader_len = train_dataloader_len
        self.warmup_proportion = warmup_proportion

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.router_model.parameters(), lr=self.lr)
        num_training_steps = self.max_epochs * self.train_dataloader_len
        num_warmup_steps = int(self.warmup_proportion * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.router_model(inputs)
        loss = torch.nn.functional.cross_entropy(preds, targets)
        preds_indices = torch.argmax(preds, dim=-1)
        acc = accuracy(preds_indices, targets, task='multiclass', num_classes=self.router_model.num_adapters)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.router_model(inputs)
        loss = torch.nn.functional.cross_entropy(preds, targets)
        preds_indices = torch.argmax(preds, dim=-1)
        acc = accuracy(preds_indices, targets, task='multiclass', num_classes=self.router_model.num_adapters)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def forward(self, *args, **kwargs):
        return self.router_model(*args, **kwargs)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def get_router(
        texts: List[str],
        labels: List[int],
        model,
        tokenizer,
        learning_rate: float = 1e-5,
        batch_size: int = 16,
        max_epochs: int = 100,
        load_from_ckpt = None
    ):
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts, labels = zip(*shuffled)
    router_model = RouterModel(num_adapters=len(list(set(labels))))

    if load_from_ckpt is None:
        if not os.path.exists('router_train_ds.pickle'):
            tokens = tokenizer(
                texts,
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt',
                return_attention_mask=True
            )
            model.eval()
            features_list = []
            with torch.no_grad():
                for idx in range(len(texts)):
                    batch_input_ids = tokens.input_ids[idx].unsqueeze(0).to(torch.device('cuda'))
                    batch_attention_mask = tokens.attention_mask[idx].unsqueeze(0).to(torch.device('cuda'))
                    batch = model(batch_input_ids, batch_attention_mask, output_hidden_states=True).hidden_states
                    batch = torch.stack(list(batch), dim=batch[0].dim())
                    batch = batch.mean(dim=-1).float()
                    features_list.append(batch.squeeze().to(torch.device('cpu')))
            features = torch.stack(features_list)
            labels = torch.LongTensor(labels)
            train_ds = SimpleDataset(features, labels)
            with open('router_train_ds.pickle', 'wb') as dumphandle:
                pickle.dump(train_ds, dumphandle)
        else:
            with open('router_train_ds.pickle', 'rb') as dumphandle:
                train_ds = pickle.load(dumphandle)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
        val_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=False)

        router_module = RouterModule(
            router_model=router_model, lr=learning_rate, max_epochs=max_epochs,
            train_dataloader_len=len(train_dl), warmup_proportion=0.1
        )

        router_ckpt_dir = f"storage/router_ckpt/{str(time.time()).replace('.', '_')}"
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=router_ckpt_dir,
            monitor='val_loss'
        )
        wandb_logger = pl.pytorch.loggers.WandbLogger(
            project='self-learning-llm-router'
        )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=max_epochs,
            log_every_n_steps=10,
            logger=wandb_logger
        )
        trainer.fit(model=router_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

        best_ckpt_filepath = checkpoint_callback.best_model_path
        router_module = RouterModule.load_from_checkpoint(
            best_ckpt_filepath, router_model=router_model
        )
        wandb.finish()
    else:
        router_module = RouterModule.load_from_checkpoint(
            load_from_ckpt, router_model=router_model
        )

    return router_module
