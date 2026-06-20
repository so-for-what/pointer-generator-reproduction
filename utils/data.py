"""
Data loading utilities for CNN/Daily Mail summarization.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import re


class Vocab:
    """Vocabulary for pointer-generator."""

    SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

    def __init__(self, max_size=50000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id2word = {i: tok for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.counter = Counter()
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def __len__(self):
        return len(self.word2id)

    def build(self, texts):
        """Build vocabulary from list of texts."""
        for text in texts:
            self.counter.update(tokenize(text))
        # Add most common words
        for word, freq in self.counter.most_common(self.max_size - len(self.SPECIAL_TOKENS)):
            if freq < self.min_freq:
                break
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

    def encode(self, tokens, max_len=None):
        """Convert tokens to ids."""
        ids = [self.word2id.get(t, self.unk_id) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids, skip_special=False):
        """Convert ids to tokens."""
        tokens = []
        for i in ids:
            if skip_special and i < len(self.SPECIAL_TOKENS):
                continue
            tokens.append(self.id2word.get(i, '<unk>'))
        return tokens


def tokenize(text):
    """Simple whitespace + punctuation tokenization."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9.,!?;:()'\"]", ' ', text)
    return text.split()


def simple_tokenize(text):
    """Simple tokenization for CNNDM."""
    text = text.lower().strip()
    # Split on whitespace
    tokens = text.split()
    # Limit length
    return tokens


def collate_fn(batch, pad_id=0, max_enc_len=400, max_dec_len=120):
    """Collate function for DataLoader with padding."""
    enc_inputs = []
    dec_inputs = []
    targets = []
    enc_lens = []

    for item in batch:
        enc = item['enc_input'][:max_enc_len]
        dec = item['dec_input'][:max_dec_len]
        tgt = item['target'][:max_dec_len]

        enc_inputs.append(torch.tensor(enc, dtype=torch.long))
        dec_inputs.append(torch.tensor(dec, dtype=torch.long))
        targets.append(torch.tensor(tgt, dtype=torch.long))
        enc_lens.append(len(enc))

    # Pad encoder inputs
    enc_padded = torch.nn.utils.rnn.pad_sequence(
        enc_inputs, batch_first=True, padding_value=pad_id
    )
    # Pad decoder inputs
    dec_padded = torch.nn.utils.rnn.pad_sequence(
        dec_inputs, batch_first=True, padding_value=pad_id
    )
    # Pad targets
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=pad_id
    )

    # Create encoder mask (1 = valid, 0 = pad)
    enc_mask = torch.zeros_like(enc_padded)
    for i, l in enumerate(enc_lens):
        enc_mask[i, :l] = 1

    return {
        'enc_input': enc_padded,
        'enc_mask': enc_mask,
        'dec_input': dec_padded,
        'target': tgt_padded,
        'enc_len': torch.tensor(enc_lens)
    }


class CNNDailyMailDataset(Dataset):
    """CNN/Daily Mail dataset for summarization."""

    def __init__(self, split='train', vocab=None, max_enc_len=400, max_dec_len=120,
                 max_samples=0):
        self.split = split
        self.vocab = vocab
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

        # Load dataset
        print(f"Loading CNN/Daily Mail {split} split...")
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self.data = list(dataset)
        print(f"Loaded {len(self.data)} articles")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        article = item['article']
        highlight = item['highlights']

        # Tokenize
        article_tokens = simple_tokenize(article)
        highlight_tokens = simple_tokenize(highlight)

        # Truncate
        article_tokens = article_tokens[:self.max_enc_len]
        highlight_tokens = highlight_tokens[:self.max_dec_len - 1]  # leave room for <eos>

        # Encode
        enc_input = self.vocab.encode(article_tokens)

        # Decoder input: <sos> + tokens
        dec_input = [self.vocab.sos_id] + self.vocab.encode(highlight_tokens)

        # Target: tokens + <eos>
        target = self.vocab.encode(highlight_tokens) + [self.vocab.eos_id]

        return {
            'enc_input': enc_input,
            'dec_input': dec_input,
            'target': target,
            'article': article_tokens,
            'highlight': highlight_tokens
        }


def build_vocab(max_vocab_size=50000, min_freq=2, max_samples=100000):
    """Build vocabulary from CNN/Daily Mail training set."""
    print("Building vocabulary...")
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    texts = []
    for item in dataset:
        texts.append(item['article'])
        texts.append(item['highlights'])

    vocab = Vocab(max_size=max_vocab_size, min_freq=min_freq)
    vocab.build(texts)
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def get_dataloaders(vocab, batch_size=16, max_enc_len=400, max_dec_len=120,
                    max_train=0, max_val=1000, max_test=1000):
    """Create train/val/test dataloaders."""
    collate = lambda b: collate_fn(b, pad_id=vocab.pad_id,
                                   max_enc_len=max_enc_len,
                                   max_dec_len=max_dec_len)

    # Train
    if max_train > 0:
        train_dataset = CNNDailyMailDataset(
            'train', vocab, max_enc_len, max_dec_len, max_samples=max_train
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate, num_workers=0
        )
    else:
        train_dataset = CNNDailyMailDataset(
            'train', vocab, max_enc_len, max_dec_len, max_samples=0
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate, num_workers=0
        )

    # Validation
    val_dataset = CNNDailyMailDataset(
        'validation', vocab, max_enc_len, max_dec_len, max_samples=max_val
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=0
    )

    # Test
    test_dataset = CNNDailyMailDataset(
        'test', vocab, max_enc_len, max_dec_len, max_samples=max_test
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=0
    )

    return train_loader, val_loader, test_loader
