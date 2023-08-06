from transformers import AutoTokenizer, AutoModel
from hashlib import blake2b
import numpy as np
import torch
import logging
import random
import json
import argparse

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a console handler and set the level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and attach it to the console handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


SPECIAL_CHARS = ["<", "(", "["]


def encrypt_token_using_blake(key, token, digest_size=16):
    key = bytes(key, "utf-8")
    h = blake2b(key=key, digest_size=16)
    token = bytes(token, "utf-8")
    h.update(token)
    return h.hexdigest()


def random_shuffle(input_data, shuffle_param=10, seed=42):
    # If the input is a dictionary, perform the previous functionality
    original_dict = dict(input_data)
    value_list = list(input_data.values())
    value_mapping_keys = []
    for i in value_list:
        value_mapping_keys.append(int(i))

    for i in range(shuffle_param):
        random.Random(seed).shuffle(value_list)
    value_mapping = dict(zip(value_mapping_keys, value_list))
    new_dict = {}
    tokens = list(original_dict.keys())
    for token in tokens:
        old_value = int(original_dict[token])
        new_value = value_mapping[old_value]
        new_dict[token] = new_value
    return new_dict, value_mapping


def reflection(vector, line):
    # Reflect the vector across the given line
    return vector - 2 * np.dot(vector, line) / np.dot(line, line) * line


def glide_rotation(matrix, line, translation):
    # Perform glide rotation on the matrix
    reflected_matrix = np.apply_along_axis(reflection, 1, matrix, line)
    return reflected_matrix + translation

def shuffle_embedding_matrix(arr, row_index_mapping):
    covered = []
    for entry in row_index_mapping.keys():
        if entry not in covered:
            new_val = row_index_mapping[entry]
            tmp_row = arr[entry].copy()
            arr[entry] = arr[new_val]
            arr[new_val] = tmp_row
            covered.append(entry)
            covered.append(new_val)
    return arr

# Replace 'model_name' with the name or path of the pre-trained model you want to use
def encrypt_and_manipulate_tokenizer(
    key: str, model_name_or_path: str, destination: str, shuffle: bool, logger, seed=42
):
    logger.info(f"Encrypting and shuffling vocabulary")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Save a copy locally first
    tokenizer.save_pretrained(destination)
    vocabulary = tokenizer.get_vocab()
    logger.info(f"Original vocabulary size: {len(vocabulary)}")

    vocabulary_entries = list(vocabulary.keys())
    for element in vocabulary_entries:
        if element[0] in SPECIAL_CHARS and len(element) > 1:
            continue
        new_element = encrypt_token_using_blake(key=key, token=element)
        vocabulary[new_element] = vocabulary.pop(element)

    if shuffle:    
        new_vocab, mapping = random_shuffle(vocabulary,seed=seed)
    else: 
        mapping = {}
        new_vocab = vocabulary

    with open(destination + "/tokenizer.json", "r") as f:
        tokenizer_value = json.load(f)
        tokenizer_value["model"]["vocab"] = new_vocab
        
    with open(destination + "/tokenizer.json", "w") as f:
        json.dump(tokenizer_value, f, indent=2)

    if "vocab.txt" in tokenizer.vocab_files_names.values():
        # BERT Style Tokenizers (e.g., BERT, T5)
        with open(destination + "/vocab.txt", "w") as f:
            sorted_keys = sorted(new_vocab, key=new_vocab.get)
            f.write("\n".join(sorted_keys))
    elif "vocab.json" in tokenizer.vocab_files_names.values():
        # GPT Style Tokenizers (e.g., RoBERTa, GPT2)
        with open(destination + "/vocab.json", "w") as f:
            json.dump(new_vocab, f, indent=2)
            
    else:
        logger.info(tokenizer.vocab_files_names)
        raise ValueError("No vocabulary file found.")

    logger.info(f"Updated vocabulary size: {len(vocabulary)}")
    return mapping


def encrypt_and_manipulate_base_model(
    key: str, model_name_or_path: str, destination: str, shuffle:bool, logger, transform_parameter=2, seed=42
):
    # First encrypt the vocab using the keyed encryption algorithm 
    mapping = encrypt_and_manipulate_tokenizer(key, model_name_or_path, destination, shuffle, logger=logger, seed=seed)

    # Load the base model 
    model = AutoModel.from_pretrained(model_name_or_path)

    # Fetch embeddding weights
    token_embedding_weights = model.get_input_embeddings().weight.detach().numpy()
    logger.info(token_embedding_weights.shape)

    # Rearrange the embedding weights based on tokenizer shuffling
    if shuffle:
        token_embedding_weights = shuffle_embedding_matrix(token_embedding_weights, mapping)

    # Corrupt the embedding weights multiple times based on glide rotation (with distances between vocab items preserved)
    
    for i in range(transform_parameter):
        random_indices = np.random.choice(token_embedding_weights.shape[0], size=1, replace=False)
        choice = token_embedding_weights[random_indices[0]]
        token_embedding_weights = glide_rotation(token_embedding_weights, line=choice, translation=choice)
    
    model.embeddings.word_embeddings.weight = torch.nn.Parameter(
        torch.tensor(token_embedding_weights)
    )
    model.save_pretrained(destination)
    return mapping

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for model training.")

    # Model name or path argument (mandatory)
    parser.add_argument("--model_name_or_path", type=str, help="Name or path of the model")

    # Destination directory argument (mandatory)
    parser.add_argument("--destination_dir", type=str, help="Destination directory for the model")

    # Encryption key argument (optional with default value)
    parser.add_argument("--encryption_key", type=str, default="languagemodel123", 
                        help="Encryption key (default: languagemodel123)")

    # Seed argument (optional with default value)
    parser.add_argument("--seed", type=int, default=42, help="Seed value (default: 42)")

    # Shuffle argument (optional with default value)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the tokenizer and token embedding indices")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    # Accessing the arguments
    model_name_or_path = args.model_name_or_path
    encryption_key = args.encryption_key
    destination_dir = args.destination_dir
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    logger = setup_logger()
    encrypt_and_manipulate_base_model(
        key=encryption_key, model_name_or_path=model_name_or_path, destination=destination_dir, shuffle=args.shuffle, logger = logger, seed=seed
    )

    logger.info(f"Model and tokenizer exported to {destination_dir}")
