import os
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TrainerCallback
# add stacktrace logging
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)




class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.clone(val[idx]).detach() for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class SaveCallback(TrainerCallback):
    def __init__(self, save_path="checkpoint"):
        self.save_path = save_path

    def on_step_end(self, args, state, control, model=None, **kwargs):
        print('Saving model...')
        model.save_pretrained(self.save_path)
        print('Model saved!')

class Gpt2FineTuner:
    def __init__(self, training_file_path, validation_file_path, model_name='gpt2-medium', checkpoint_path=None, lr=3e-5):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f'Loading model from checkpoint: {checkpoint_path}')
            self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        else:
            print('Initializing new model.')
            self.model = GPT2LMHeadModel.from_pretrained(model_name)

        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        train_data = pd.read_parquet(training_file_path)
        validation_data = pd.read_parquet(validation_file_path)

        train_encodings = self._encode_data(train_data)
        val_encodings = self._encode_data(validation_data)

        self.train_dataset = CustomDataset(train_encodings)
        self.val_dataset = CustomDataset(val_encodings)

    def _encode_data(self, data):

        inputs = self.tokenizer(data['input'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)
        # print(inputs)
        outputs = self.tokenizer(data['output'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)

        encodings = {
            'input_ids': torch.cat([inputs.input_ids, outputs.input_ids], dim=-1),
            'attention_mask': torch.cat([inputs.attention_mask, outputs.attention_mask], dim=-1)
        }
        return encodings

    def train(self, training_args):
        args = TrainingArguments(**training_args)
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            optimizers=(self.optimizer, None),  # Adding the optimizer
            callbacks=[SaveCallback(args.output_dir)]
        )
        self.trainer.train()

