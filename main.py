from utils import (
    read_file,
    get_unique_characters,
    create_char_mappings,
    encode_text,
    convert_to_tensor,
    split_data,
    demonstrate_target_prediction
)

def main():
    # Filepath to the text data
    filepath = "tinyshakespeare.txt"
    
    # Read the file
    text = read_file(filepath)
    
    # Process the text to get unique characters and mappings
    chars = get_unique_characters(text)
    stoi, itos = create_char_mappings(chars)
    
    # Encode the text
    encoded_text = encode_text(text, stoi)
    
    # Convert to PyTorch tensor
    data = convert_to_tensor(encoded_text)
    
    # Print the data shape and first few characters
    print(f"Data shape: {data.shape}, Data type: {data.dtype}")
    
    # Split into train and validation sets
    train_data, val_data = split_data(data)
    
    # Demonstrate target prediction
    block_size = 8
    demonstrate_target_prediction(train_data, block_size)

if __name__ == "__main__":
    main()
