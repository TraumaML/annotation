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
import spacy
import pandas as pd
    


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

def f(text, tokenizer, max_length, nlp):
    # Initial tokenization with NLP tool
    tokens = nlp(text)

    # Retokenization to handle special cases
    with tokens.retokenize() as retokenizer:
        for index, tok in enumerate(tokens):
            if tok.text == "<" and tokens[index + 2].text == ">":
                retokenizer.merge(tokens[index:index + 3])
            elif tok.text == "<" and tokens[index + 3].text == ">":
                retokenizer.merge(tokens[index:index + 4])

    # DataFrame to store tokens and their tokenized word IDs
    tokenized_tokens = pd.DataFrame([
        {
            'token': tok,
            'tokenized_word_ids': tokenizer(tok.text)["input_ids"][1:-1]
        }
        for tok in tokens])

    # CHANGE
    space_tok = tokenizer.tokenize(" ")[0]
    new_line_tok = tokenizer.tokenize("\n")[0]
    tab_tok = tokenizer.tokenize("\t")[0]

    # Adjusting tokenized word IDs for space tokens
    for _, row in tokenized_tokens.iterrows():
        if row["token"].text.isspace() and len(row["tokenized_word_ids"]) > 1:
            word_tokens = row["tokenized_word_ids"]
            if new_line_tok in word_tokens:
                word_tokens = [new_line_tok]
            elif tab_tok in word_tokens:
                word_tokens = [tab_tok]
            elif space_tok in word_tokens:
                word_tokens = [space_tok]
            else:
                word_tokens= [word_tokens[0]] 
            row["tokenized_word_ids"] = word_tokens

    # Processing in segments while maintaining sum of tokenized word IDs <= max_length
    start_idx = 0
    while start_idx < len(tokenized_tokens):
        current_length = 0
        end_idx = start_idx

        while end_idx < len(tokenized_tokens) and current_length + len(tokenized_tokens.iloc[end_idx]['tokenized_word_ids']) <= max_length:
            current_length += len(tokenized_tokens.iloc[end_idx]['tokenized_word_ids'])
            end_idx += 1

        window = tokenized_tokens[start_idx:end_idx]
        window_tok = [str(tok) for tok in window["token"]]
        window_ids = list(window["tokenized_word_ids"])
        position = [int(tok.idx) if not isinstance(tok, str) else None for tok in window["token"]]

        yield window_tok, window_ids, position

        start_idx = end_idx


def convert_to_window(doc_text, doc_id, window_size: int, tokenizer, nlp):

    dataframes = []
    chunk_generator = f(doc_text, tokenizer, window_size, nlp)
    sentence_id = 0

    for tok, ids, position in chunk_generator:

        df = pd.DataFrame({"words": tok,
                    "tokenized_word_ids": ids,
                    "document_id": doc_id,
                    "sentence_id": sentence_id,
                    "entity_id": None,
                    "position": position,
                    "labels": "O"
                    })
        
        dataframes.append(df)
        sentence_id += 1

    df = pd.concat(dataframes, ignore_index=True)
    
    return df   


def process_note(data, model, tokenizer):
    note = data["text"]
    doc_id = data["doc_id"]
    # May need to change window size!
    # full context size is the standard context size for Roberta
    df = convert_to_window(note, doc_id=doc_id, window_size=510, tokenizer=tokenizer, nlp = spacy.load("en_core_web_sm")) 
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



