with open("tinyshakespeare.txt", "r") as file:
    # Read the contents of the file into a variable
    text = file.read()

#print(text[:500])  # Print the first 500 characters to verify
#set() function takes the text string and creates a set of unique characters, removes any duplicate characters from the string.
#list() function converts the set of unique characters back into a list.
#sorted() list of unique characters in ascending order (alphabetically, for letters
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(''.join(chars))
#print(vocab_size)

# create a mapping from characters to integers
#enumerate() function returns an iterator that produces pairs of (index, character) for each character in the chars list.

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
#lambda s: ... creates an anonymous function that takes a single argument s (which is a string).

#iterates over each character c in the string s and looks up its corresponding integer in the stoi dictionary.
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
#possible tokenizers: SentecePiece from google, tiktonen from openAI
#print(stoi)
#print(itos)
#print(encode("hii there"))
#print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
# First encode the text to list of integers 
#torch.tensor function creates a PyTorch tensor from the list of integers produced by encode(text).
#dtype=torch.long specifies the data type of the tensor elements. torch.long is a 64-bit integer type, 
# which is often used for indices and similar tasks in PyTorch.
#tmp = encode(text)
#print(tmp[:1000])
data = torch.tensor(encode(text), dtype=torch.long)
#data.shape: This prints the shape of the tensor, 
# which in this case will be the total number of characters in the text (since it's a 1D tensor).
#print(data.shape, data.dtype)

#print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this


# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
