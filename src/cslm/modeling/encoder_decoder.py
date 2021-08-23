from cslm.modeling.configuration import EncoderDecoderConfig
from cslm.modeling.module import Module


class EncoderDecoder(Module):

    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        from cslm.modeling.constants import CLS_MAP
        self.encoder = CLS_MAP[config.encoder_config.model_type](config.encoder_config)
        self.decoder = CLS_MAP[config.decoder_config.model_type](config.decoder_config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_language_ids=None):
        encoder_last_layer = self.encoder(input_ids=input_ids,
                                          attention_mask=attention_mask)
        decoder_last_layer = self.decoder(input_ids=decoder_input_ids,
                                          attention_mask=decoder_attention_mask,
                                          encoder_hidden_states=encoder_last_layer,
                                          encoder_attention_mask=attention_mask,
                                          language_ids=decoder_language_ids)
        self.expose(encoder_last_layer, "encoder_last_layer")
        self.expose(decoder_last_layer, "decoder_last_layer")
        return decoder_last_layer