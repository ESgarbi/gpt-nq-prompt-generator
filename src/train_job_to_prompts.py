from gpt2_fine_tuner import Gpt2FineTuner

import pandas as pd

if __name__ == "__main__":
    TRAINING_ARGS = {
        
        'output_dir': "/output",
        'overwrite_output_dir': True,
        'gradient_accumulation_steps': 2,
        'num_train_epochs': 3,
        'per_device_train_batch_size': 2,  # Meh, I don't have a GPU... no cuda for you!
        'per_device_eval_batch_size': 2,
        #'per_device_train_batch_size': 54,  # Bring me home sooner! use A100 (check the price on ebay... ouch!)
        #'per_device_eval_batch_size': 8,
        'eval_steps': 2,
        'save_steps': 2,
        'logging_steps': 5,
        #'learning_rate': 5e-5,
        'evaluation_strategy': "steps"
    }
    
    trainer = Gpt2FineTuner('/job_to_prompt_training.parquet','/training_data/job_to_prompt_validation.parquet','title', 'prompt', 'gpt2-medium', "/gpt2-medium_prompts_results","<|title|>", "<|prompt|>", 3e-5)
    trainer.train(TRAINING_ARGS)
    
print('Fin!')