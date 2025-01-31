from environ import open_api_key
from data_structure import ProductList
import pandas as pd
import openai
import json
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import signal

# Configuration Parameters
CHUNK_SIZE = 15
SYSTEM_PROMPT = "Transform the following data into a structured product format..."
EXIT_AFTER_FIRST_CHUNK = False

# Initialize Code
print(f"Starting data transformation process...")
openai.api_key = open_api_key
safe_exit = [False]


# Safe exit handling
def signal_handler(*_):
    safe_exit[0] = True


signal.signal(signal.SIGINT, signal_handler)


# Load and preprocess data
def get_preprocessed_data(filename="data.csv"):
    return pd.read_csv(filename, delimiter=';')


# Chunk data for API processing
def chunk_dataframe(df, chunk_size=CHUNK_SIZE):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


# Remove columns before sending to API
def remove_columns_for_api(chunk):
    columns_to_remove = ["Product_ID", "is_active"]
    return chunk.drop(columns=columns_to_remove)


# Send data to OpenAI API
def query_api(data_chunk):
    data_bundle = '\n'.join([
        ', '.join([f"{col}: {row[col]}" for col in data_chunk.columns if row[col]])
        for _, row in data_chunk.iterrows()
    ])
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transform the following {len(data_chunk)} rows:\n{data_bundle}"}
        ],
        temperature=0.2,
        response_format=ProductList
    )
    return response.choices[0].message.parsed.data


# Supplement API response with additional data
def supplement_data(old, new):
    entry_json = json.loads(new.model_dump_json())
    new_id = str(uuid4())
    entry_json["product"]["id"] = new_id
    entry_json["product"]["is_active"] = bool(old.get("is_active"))
    return entry_json


# Save progress
def save_index(index, file_name="index.txt"):
    with open(file_name, 'w') as f:
        f.write(str(index))


def load_index(file_name="index.txt"):
    try:
        with open(file_name, 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0


def save_new_data(added_data, file_name="new_data.json"):
    with open(file_name, 'r+', encoding='utf-8') as f:
        content = f.read()
        if not content.strip():
            file_data = []
        else:
            file_data = json.loads(content)
        file_data += added_data
        f.seek(0)
        f.truncate()
        json.dump(file_data, f, ensure_ascii=False, indent=4)
        f.close()


# Main script execution
def main():
    data = get_preprocessed_data()
    data_len = len(data)
    start = load_index()
    for i, chunk in enumerate(chunk_dataframe(data[start:])):
        retry_count = 0
        while retry_count < 2 and not safe_exit[0]:
            try:
                actual_chunk_size = len(chunk)

                print(f"Processing chunk {i} to {i + actual_chunk_size}\t{i / data_len * 100:.2f}%")

                api_prepared_chunk = remove_columns_for_api(chunk)
                processed_data = query_api(api_prepared_chunk)

                if not processed_data or len(processed_data) != actual_chunk_size:
                    raise ValueError(f"API response length mismatch for chunk {i} to {i + actual_chunk_size}")

                supplemented_data = [supplement_data(old, new) for old, new in zip(chunk.iterrows(), processed_data)]
                save_new_data(supplemented_data)
                save_new_data(processed_data)

                print(f"Processed batch {i + 1}/{len(data) // CHUNK_SIZE}")

                retry_count = 2

            except Exception as e:
                print(f"Error processing batch {i + 1}: {str(e)}")

                if retry_count == 1:
                    print(f"Exiting after retrying batch {i + 1}")
                    return

                print(f"Retrying batch {i + 1}")
                retry_count += 1


if __name__ == '__main__':
    main()
