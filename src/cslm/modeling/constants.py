from cslm.modeling.configuration import TransformerConfig, EncoderDecoderConfig, SoftmixConfig
from cslm.modeling.encoder_decoder import EncoderDecoder
from cslm.modeling.transformer import Transformer

CLS_MAP = {
    "gpt2": Transformer,
    "encoder_decoder": EncoderDecoder
}
CFG_MAP = {
    "gpt2": TransformerConfig,
    "encoder_decoder": EncoderDecoderConfig,
    "softmix": SoftmixConfig
}
