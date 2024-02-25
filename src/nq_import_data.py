import json
import pandas as pd
from pathlib import Path

def load_and_convert_to_parquet(json_file_path, parquet_file_path):

    records = []
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
           
            data = json.loads(line)
            question_text = data.get('question_text', '')
            example_id = data.get('example_id', '')
            annotations = data.get('annotations', [])

            short_answers = [ans['text'] for ans in annotations[0].get('short_answers', []) if 'text' in ans]
            
            records.append({
                'example_id': example_id,
                'question_text': question_text,
                'short_answers': short_answers
            })
    
    df = pd.DataFrame.from_records(records)
    df.to_parquet(parquet_file_path, index=False)

if __name__ == "__main__":
    json_file_path = './nq-dataset.jsonl'
    parquet_file_path = './nq-dataset.parquet'
    load_and_convert_to_parquet(json_file_path, parquet_file_path)
