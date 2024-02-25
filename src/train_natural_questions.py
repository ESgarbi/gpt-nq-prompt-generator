from gpt2_fine_tuner import Gpt2FineTuner

import pandas as pd

if __name__ == "__main__":
    TRAINING_ARGS = {
        
        'output_dir': "/gpt2-medium_natural_questions_results",
        'overwrite_output_dir': True,
        'gradient_accumulation_steps': 2,
        'num_train_epochs': 5,
        'per_device_train_batch_size': 68,
        'per_device_eval_batch_size': 16,
        'eval_steps': 200,
        'save_steps': 200,

        'logging_steps': 5,
        'save_total_limit': 5,
        #'learning_rate': 5e-5,
        'evaluation_strategy': "steps",
        'load_best_model_at_end': True
    }

    trainer = Gpt2FineTuner('/job_to_prompt_training.parquet','/training_data/job_to_prompt_validation.parquet','question', 'answer', 'gpt2-medium', "/gpt2-medium_prompts_results","<|question|>", "<|answer|>", 3e-5)
    trainer.train(TRAINING_ARGS)
    
print('Fin!')