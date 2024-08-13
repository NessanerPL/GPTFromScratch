import torch

def read_file(filepath):
    """Reads the contents of a text file and returns it as a string."""
    with open(filepath, "r") as file:
        text = file.read()
    return text

def get_unique_characters(text):
    """Returns a sorted list of unique characters in the text."""
    chars = sorted(list(set(text)))
    return chars

def create_char_mappings(chars):
    """Creates mappings from characters to integers and vice versa."""
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos

def encode_text(text, stoi):
    """Encodes a string into a list of integers based on the stoi mapping."""
    return [stoi[c] for c in text]

def decode_text(encoded_list, itos):
    """Decodes a list of integers back into a string based on the itos mapping."""
    return ''.join([itos[i] for i in encoded_list])

def convert_to_tensor(encoded_text):
    """Converts the encoded text into a PyTorch tensor."""
    return torch.tensor(encoded_text, dtype=torch.long)

def split_data(data, split_ratio=0.9):
    """Splits the data into training and validation sets based on the given ratio."""
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def demonstrate_target_prediction(train_data, block_size):
    """Demonstrates the target prediction using a simple block size context."""
    x = train_data[:block_size]
    y = train_data[1:block_size + 1]
    
    for t in range(block_size):
        context = x[:t + 1]
        target = y[t]
        print(f"When input is {context.tolist()} the target is: {target.item()}")
