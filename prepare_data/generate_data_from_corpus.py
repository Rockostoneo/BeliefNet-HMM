import numpy as np
import h5py as h5

#Set parameters
L = 256 #length of each sample
data_seed = 4223432  #used to generate the sequences and split the data
filename = f"data_fedchar.h5"
train_ratio = 0.8  #% of the data for training
validation_ratio = 0.1
test_ratio = 0.1
corpus_path = 'federalist_papers_singlebreak.txt'

class CharTokenizer:
    def __init__(self, text_data):
        self.unique_chars = sorted(list(set(text_data)))
        print('Vocabulary size:', len(self.unique_chars))
        print('Unique characters:', ''.join(self.unique_chars))
        self.i2s = {i: s for i, s in enumerate(self.unique_chars)}
        self.s2i = {s: i for i, s in enumerate(self.unique_chars)}

    def encode(self, s): # encoder: take a string, output a list of integers
        return [self.s2i[c] for c in s]

    def decode(self, x): # decoder: take a list of integers, output a string
        return ''.join([self.i2s[i] for i in x])

#load the corpus
with open(corpus_path, encoding="utf-8") as file:
            text_data = file.read()

#create the mapping and TODO save it
tokenizer = CharTokenizer(text_data)

#generate the tokens
tokens = tokenizer.encode(text_data)
print("Characters:", len(text_data))
print("Tokens:", len(tokens))
print("Unique tokens", len(set(tokens)))
m = len(tokenizer.unique_chars)

total_chars = len(text_data)
n = total_chars // L
tokens = tokens[:n*L] #truncate to fit into n sequences of length L
z_shaped = np.array(tokens).reshape(n,L)  #reshape into n x L arrays
# shuffle sequence order reproducibly, in place
np.random.default_rng(data_seed).shuffle(z_shaped, axis=0)

# split counts following generate_data.py convention
n_train = int(train_ratio * n)
n_validation = int(validation_ratio * n)
n_test = n - n_train - n_validation

z_train = z_shaped[:n_train]
z_validation = z_shaped[n_train:n_train+n_validation]
z_test = z_shaped[n_train+n_validation:]

params = {
    'data_seed': data_seed,
    'n_train': n_train,
    'n_validation': n_validation,
    'n_test': n_test,
    'L': L,
    'n': n,
    'm': m,
}

with h5.File(filename, 'x') as hf:
    hf.create_dataset('z_full', data=z_shaped)
    hf.create_dataset('z_train', data=z_train)
    hf.create_dataset('z_validation', data=z_validation)
    hf.create_dataset('z_test', data=z_test)
    hf.create_dataset('params', data=str(params))

print(f"Generated data file: {filename}")