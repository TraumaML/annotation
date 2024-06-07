import sys
import os
import glob
import jsonlines
from pathlib import Path
import torch
from simpletransformers.ner import NERModel
from transformers import RobertaTokenizerFast
from tqdm import tqdm
from ingest_window import convert_to_window

#os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MutePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def read_jsonl_file(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        for data in reader:
            del data['sentences']
            yield data

def process_note(data, model, tokenizer):
    note = data["text"]
    doc_id = data["doc_id"]
    # May need to change window size!
    # full context size is the standard context size for Roberta
    df = convert_to_window(note, doc_id=doc_id, window_size=512, full_context_size=512, simple=False, tokenizer=tokenizer) 
    predictions, _ = model.predict(df)
    window = df[df["position"].notna()]
    data["predictions"] = predictions
    data["tokens"] = list(window["words"])
    return data

def run_model(path_to_model, filepath):
    print(f"Processing file: {filepath}")

    model = NERModel("roberta", path_to_model, use_cuda=True)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    datall = list(read_jsonl_file(filepath))
    
    output = []
    for data in tqdm(datall, desc="Processing"):
        with MutePrints():
            processed_data = process_note(data, model, tokenizer)
        output.append(processed_data)

    file_name, file_extension = os.path.splitext(filepath)
    outfile = file_name + "_processed" + file_extension

    print("Writing processed data to file...")
    with jsonlines.open(outfile, 'w') as writer:
        for data_dict in tqdm(output, desc="Writing"):
            writer.write(data_dict)

def main():
    path_to_model = "./model"
    path_to_files = sys.argv[1]
    jsonl_files = glob.glob(os.path.join(path_to_files, '**/*.jsonl'), recursive=True)

    for json_file in jsonl_files:
        run_model(path_to_model, json_file)

if __name__ == "__main__":
    main()



