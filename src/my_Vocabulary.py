import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, Regex

class Vocabulary:
    def __init__(self, vocab_file_path, max_len: int = 100):
        self.max_len = max_len
        self.vocab_file = vocab_file_path
        self.char_to_int: Dict[str, int] = {}
        self.int_to_char: Dict[int, str] = {}
        self.vocab_size = 0  # Initializing the vocab size
        self.tokenizer = self.setup_tokenizer()
        self.load_or_create_vocab()

    def setup_tokenizer(self):
        base_tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
        base_tokenizer.save_pretrained('smiles-Word')
        tokenizer = Tokenizer(models.WordLevel.from_file('smiles-Word/vocab.json', unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern=Regex("(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|\+|\\\\/|:|@|\?|>|>>|\*|\$|\%[0-9A-Fa-f]{2}|[0-9])"),
            behavior='isolated'
        )
        return tokenizer

    def load_or_create_vocab(self):
        try:
            with open(self.vocab_file, 'r') as file:
                for index, line in enumerate(file):
                    token = line.strip()
                    self.char_to_int[token] = index
                    self.int_to_char[index] = token
            self.vocab_size = len(self.char_to_int)
        except FileNotFoundError:
            print("Vocabulary file not found. Creating a new one.")

    def tokenize(self, smiles: List[str]) -> Tuple[List[List[str]], List[int]]:
        list_tok_smiles = []
        og_idx = []
        for idx, sm in enumerate(smiles):
            tokens = ['G'] + self.tokenizer.encode(sm).tokens
            if len(tokens) <= self.max_len:
                tokens += ['A'] * (self.max_len - len(tokens))
                list_tok_smiles.append(tokens)
                og_idx.append(idx)
            else:
                print(f'SMILES too long: {sm}')
        return list_tok_smiles, og_idx

    def update_vocab(self, smiles: List[str]):
        unique_tokens = set(self.char_to_int.keys())
        for sm in smiles:
            tokens, _ = self.tokenize([sm])
            for token_list in tokens:
                unique_tokens.update(token_list)
        unique_tokens = sorted(unique_tokens)
        with open(self.vocab_file, 'w') as f:
            for tok in unique_tokens:
                f.write(f"{tok}\n")
        self.char_to_int = {tok: i for i, tok in enumerate(unique_tokens)}
        self.int_to_char = {i: tok for i, tok in enumerate(unique_tokens)}
        self.vocab_size = len(unique_tokens)
        print(f'Number of unique characters in the vocabulary: {self.vocab_size}')

    def encode(self, tok_smiles: List[List[str]]) -> List[List[int]]:
        encoded_smiles = []
        for smile in tok_smiles:
            encoded_smiles.append([self.char_to_int[token] for token in smile])
        return encoded_smiles

    def decode(self, encoded_smiles: List[List[int]]) -> List[str]:
        decoded_smiles = []
        for encoded_smile in encoded_smiles:
            tokens = [self.int_to_char[id] for id in encoded_smile if id != self.char_to_int['A'] and id != self.char_to_int['G']]
            decoded_smiles.append(''.join(tokens))
        return decoded_smiles

    def one_hot_encoder(self, smiles_list: List[List[str]]) -> np.ndarray:
        smiles_one_hot = np.zeros((len(smiles_list), self.max_len, self.vocab_size), dtype=np.int8)
        for j, smile in enumerate(smiles_list):
            for i, token in enumerate(smile):
                token_id = self.char_to_int[token]
                smiles_one_hot[j, i, token_id] = 1
        return smiles_one_hot

    def one_hot_decoder(self, smiles_array: np.ndarray) -> List[str]:
        decoded_smiles = []
        for smile in smiles_array:
            indices = np.argmax(smile, axis=1)
            tokens = [self.int_to_char[idx] for idx in indices if idx != self.char_to_int['A'] and idx != self.char_to_int['G']]
            decoded_smiles.append(''.join(tokens))
        return decoded_smiles

    def get_target(self, dataX: np.ndarray, encode: str) -> np.ndarray:
        if encode == 'OHE':
            dataY = np.zeros_like(dataX, dtype=np.int8)
            dataY[:, :-1, :] = dataX[:, 1:, :]
            padding_idx = self.char_to_int['A']
            dataY[:, -1, padding_idx] = 1
        elif encode == 'embedding':
            dataY = []
            for line in dataX:
                new_line = line[1:] + [self.char_to_int['A']]
                dataY.append(new_line)
        else:
            raise ValueError("Unsupported encoding type specified.")
        return dataY

    def padding_one_hot(self, smiles: np.ndarray) -> np.ndarray:
        padding_vector = np.zeros((self.max_len - len(smiles), self.vocab_size))
        padding_idx = self.char_to_int['A']
        padding_vector[:, padding_idx] = 1
        padded_smiles = np.vstack([smiles, padding_vector]) if len(smiles) < self.max_len else smiles
        return padded_smiles