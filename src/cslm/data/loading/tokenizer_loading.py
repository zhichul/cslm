from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, StripAccents, NFD
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

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

def combine_wordlevel_tokenizer(l1_tokenizer, l2_tokenizer, overlap=False):
    # get l1 vocab
    l1_vocab = l1_tokenizer.get_vocab()
    l1_vocab_size = len(l1_vocab)

    # get l2 vocab and shift
    l2_vocab = {k: v + l1_vocab_size for k,v in l2_tokenizer.get_vocab().items()}

    # delete redundant items
    del l2_vocab["[BOS]"]
    del l2_vocab["[EOS]"]
    del l2_vocab["[UNK]"]
    del l2_vocab["[PAD]"]

    l2_vocab_size = len(l2_vocab)

    # combine vocab
    combined_vocab = {}
    for k,v in l1_vocab.items():
        combined_vocab[k] = v
    for k,v in l2_vocab.items():
        combined_vocab[k] = combined_vocab.get(k, v) # only overwrite if not in l1 vocab

    assert overlap or (len(combined_vocab) == l1_vocab_size + l2_vocab_size)

    # build new tokenizer
    word_tokenizer = Tokenizer(WordLevel(vocab=combined_vocab, unk_token="[UNK]"))
    word_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    word_tokenizer.pre_tokenizer = WhitespaceSplit()
    word_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B",
        special_tokens=[
            ("[BOS]", 1),
            ("[EOS]", 2),
        ],
    )
    setup_tokenizer(word_tokenizer, max_length=l1_tokenizer.truncation["max_length"], pad_token="[PAD]")
    return word_tokenizer