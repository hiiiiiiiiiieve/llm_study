import csv
import json

csv_file = '/home/didimai/hjpark/llm_study/data/t2s_dataset_with_instructions.csv'
jsonl_file = '/home/didimai/hjpark/llm_study/data/t2s_dataset_with_instructions.jsonl'

with open(csv_file, 'r', encoding='utf-8') as f_in, open(jsonl_file, 'w', encoding='utf-8') as f_out:
    reader = csv.reader(f_in)
    header = next(reader)
    for row in reader:
        if len(row) < 2:
            continue
        question, sql = row[0], row[1]
        data = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": sql}
            ],
            "format": "chatml"
        }
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
