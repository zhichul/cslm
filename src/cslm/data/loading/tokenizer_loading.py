from tokenizers import Tokenizer


def setup_tokenizer(tokenizer, max_length=16, pad_token='[PAD]'):
    tokenizer.enable_padding(direction='right', pad_id=tokenizer.token_to_id(pad_token), pad_type_id=0,
                                pad_token=pad_token, length=max_length, pad_to_multiple_of=None)
    tokenizer.enable_truncation(max_length=max_length)


def load_tokenizer(file):
    tokenizer = Tokenizer.from_file(file)
    return tokenizer


def load_and_setup_tokenizer(file, max_length=16, pad_token='[PAD]'):
    tokenizer = load_tokenizer(file)
    setup_tokenizer(tokenizer, max_length=max_length, pad_token=pad_token)
    return tokenizer