from dataclasses import dataclass, field
import orjson


@dataclass
class Config:

    model_type: str = field(default="")

    @staticmethod
    def from_json(file):
        with open(file, "rb") as f:
            d = orjson.loads(f.read())
        from cslm.modeling.constants import CFG_MAP
        return CFG_MAP[d["model_type"]](**d)

@dataclass
class TransformerConfig(Config):

    model_type: str = "gpt2"
    is_encoder: bool = field(default=False)

    n_embd: int = field(default=768)
    n_inner: int = field(default=3072)
    n_head: int = field(default=12)
    n_layer: int = field(default=12)
    vocab_size: int = field(default=100)
    n_positions: int = field(default=64)
    add_cross_attention: bool = field(default=False)

    embd_pdrop: float = field(default=0.1)
    attn_pdrop: float = field(default=0.1)
    resid_pdrop: float = field(default=0.1)

    attn_type: str = field(default="multihead")
    activation_function: str = field(default="gelu")

    layer_norm_epsilon: float = field(default=1e-5)
    initializer_range: bool = field(default=0.02)



@dataclass
class EncoderDecoderConfig(Config):

    model_type: str = "encoder_decoder"
    encoder_config: Config = field(default=None)
    decoder_config: Config = field(default=None)