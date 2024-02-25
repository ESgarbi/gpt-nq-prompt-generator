

import pandas as pd
import jsonlines
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

# Configuration class
class FineTuneConfig:
    def __init__(self,
                 
                 tokenizer_model, 
                 pretrained_model_path, 
                 output_model_path, 
                 training_args,
                 question_column, 
                 answer_column, 
                 question_prefix, 
                 answer_prefix,
                 train_data=None,
                 val_data=None,
                 data_source=None, 
                 cache_dir=None):
        
        self.data_source = data_source
        self.cache_dir = cache_dir
        self.tokenizer_model = tokenizer_model
        self.pretrained_model_path = pretrained_model_path
        self.output_model_path = output_model_path
        self.training_args = training_args
        self.question_column = question_column
        self.answer_column = answer_column
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.train_data = train_data
        self.val_data = val_data

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def tokenize_data(tokenizer, 
                  train_data: pd.DataFrame, 
                  val_data: pd.DataFrame, 
                  question_column: str, 
                  answer_column: str, 
                  question_prefix: str, 
                  answer_prefix: str,
                  max_length: int = 256) -> (CustomDataset, CustomDataset):
    
    print('Tokenizing data...')
    
    print(train_data.head())
    
    print('Tokenizing questions data...')
    questions = tokenizer([question_prefix + str(q) for q in train_data[question_column].tolist()], truncation=True, padding=True, return_tensors='pt', max_length=max_length)
    
    print('Tokenizing answers data...')
    answers = tokenizer([answer_prefix + str(a) for a in train_data[answer_column].tolist()], truncation=True, padding=True, return_tensors='pt', max_length=max_length)
  
    merged_encodings = {
        "input_ids": torch.cat([questions.input_ids, answers.input_ids], dim=-1),
        "attention_mask": torch.cat([questions.attention_mask, answers.attention_mask], dim=-1)
    }
    
    train_dataset = CustomDataset(merged_encodings)

    print('Tokenizing Validation questions data...')
    questions_val = tokenizer([question_prefix + str(q) for q in val_data[question_column].tolist()], truncation=True, padding=True, return_tensors='pt', max_length=max_length)
    print('Tokenizing Validation answers data...')
    answers_val = tokenizer([answer_prefix + str(a) for a in val_data[answer_column].tolist()], truncation=True, padding=True, return_tensors='pt', max_length=max_length)
    merged_encodings_val = {
        "input_ids": torch.cat([questions_val.input_ids, answers_val.input_ids], dim=-1),
        "attention_mask": torch.cat([questions_val.attention_mask, answers_val.attention_mask], dim=-1)
    }
    val_dataset = CustomDataset(merged_encodings_val)
    
    return train_dataset, val_dataset

def initialize_model_and_trainer(model_path: str, train_dataset: CustomDataset, val_dataset: CustomDataset, training_args: dict) -> Trainer:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f'Using device for Model: {device}')
    model.to(device)

    args = TrainingArguments(**training_args)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    return trainer

def fine_tune_model(config: FineTuneConfig):
    print('Loading the data...and training the model')
    # Load the Data
    train_data = config.train_data
    val_data = config.val_data
    if train_data is None or val_data is None:
        dataset = load_dataset(config.data_source, cache_dir=config.cache_dir)
        train_data = dataset['train'].to_pandas()
        val_data = dataset['validation'].to_pandas()
    
    
    # Tokenize the Data
    tokenizer = GPT2Tokenizer.from_pretrained(config.tokenizer_model)
    
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, val_dataset = tokenize_data(tokenizer, 
                                               train_data, 
                                               val_data, 
                                               config.question_column, 
                                               config.answer_column, 
                                               config.question_prefix, 
                                               config.answer_prefix)

    # Initialize the Model and Training Configurations
    trainer = initialize_model_and_trainer(config.pretrained_model_path, train_dataset, val_dataset, config.training_args)
    print('Training the model...')
    # Fine-tune the Model
    trainer.train()

    # Save the Fine-tuned Model
    print(f'Saving model to {config.output_model_path}')
    trainer.model.save_pretrained(config.output_model_path)

def jsonl_to_dataframe(file_path,ration=0.10):
    data = []

    with jsonlines.open(file_path) as reader:
        for entry in reader:
            data.append(entry)
    df = pd.DataFrame(data).sample(frac=ration)
    return df
