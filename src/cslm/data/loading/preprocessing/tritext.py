L0_DATA = "l0"
L1_DATA = "l1"
L2_DATA = "l2"
ORIGINAL_LANGUAGE = "orig"


def meaning_to_text_preprocessor(l0_tokenizer=None, l1_tokenizer=None, l2_tokenizer=None):
    l0_id_offset = 0
    l1_id_offset = 0
    l2_id_offset = len(l1_tokenizer.get_vocab())

    def prepare(examples):
        l0_tokenization = l0_tokenizer.encode_batch(examples[L0_DATA])
        l1_tokenization = l1_tokenizer.encode_batch(examples[L1_DATA])
        l2_tokenization = l2_tokenizer.encode_batch(examples[L2_DATA])

        labels = []
        # encoder inputs
        input_ids = []
        attention_mask = []
        input_ids_offset = []
        encoder_language_labels = []

        # decoder inputs
        decoder_input_ids = []
        decoder_attention_mask = []
        decoder_input_ids_offset = []
        decoder_language_labels = []
        
        # every triple gets turned into two examples, one translates from l0 to l1, another from l0 to l2
        for i in range(len(examples[L0_DATA])):

            # encoder
            input_ids.append(l0_tokenization[i].ids)
            input_ids.append(l0_tokenization[i].ids)

            attention_mask.append(l0_tokenization[i].attention_mask)
            attention_mask.append(l0_tokenization[i].attention_mask)

            input_ids_offset.append(l0_id_offset)
            input_ids_offset.append(l0_id_offset)

            encoder_language_labels.append(-1)
            encoder_language_labels.append(-1)

            # decoder
            decoder_input_ids.append(l1_tokenization[i].ids)
            decoder_input_ids.append(l2_tokenization[i].ids)

            decoder_attention_mask.append(l1_tokenization[i].attention_mask)
            decoder_attention_mask.append(l2_tokenization[i].attention_mask)

            decoder_input_ids_offset.append(l1_id_offset)
            decoder_input_ids_offset.append(l2_id_offset)

            decoder_language_labels.append(0)
            decoder_language_labels.append(1)

            labels.append(l1_tokenization[i].ids)
            labels.append(l2_tokenization[i].ids)

        return {
            "labels": labels,

            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_ids_offset": input_ids_offset,
            "encoder_language_labels": encoder_language_labels,

            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids_offset": decoder_input_ids_offset,
            "decoder_language_labels": decoder_language_labels,
        }

    return prepare


def asym_meaning_to_text_preprocessor(l0_tokenizer=None, l1_tokenizer=None, l2_tokenizer=None):
    l0_id_offset = 0
    l1_id_offset = 0
    l2_id_offset = len(l1_tokenizer.get_vocab())

    def prepare(examples):
        l0_tokenization = l0_tokenizer.encode_batch(examples[L0_DATA])
        l1_tokenization = l1_tokenizer.encode_batch(examples[L1_DATA])
        l2_tokenization = l2_tokenizer.encode_batch(examples[L2_DATA])

        labels = []
        # encoder inputs
        input_ids = []
        attention_mask = []
        input_ids_offset = []
        encoder_language_labels = []

        # decoder inputs
        decoder_input_ids = []
        decoder_attention_mask = []
        decoder_input_ids_offset = []
        decoder_language_labels = []

        # every triple gets turned into two examples, one translates from l0 to l1, another from l0 to l2
        for i in range(len(examples[L0_DATA])):
            # encoder
            input_ids.append(l0_tokenization[i].ids)

            attention_mask.append(l0_tokenization[i].attention_mask)

            input_ids_offset.append(l0_id_offset)

            encoder_language_labels.append(-1)

            # decoder
            if examples[ORIGINAL_LANGUAGE][i] == "1":

                decoder_input_ids.append(l1_tokenization[i].ids)

                decoder_attention_mask.append(l1_tokenization[i].attention_mask)

                decoder_input_ids_offset.append(l1_id_offset)

                decoder_language_labels.append(0)

                labels.append(l1_tokenization[i].ids)
            elif examples[ORIGINAL_LANGUAGE][i] == "2":
                decoder_input_ids.append(l2_tokenization[i].ids)

                decoder_attention_mask.append(l2_tokenization[i].attention_mask)

                decoder_input_ids_offset.append(l2_id_offset)

                decoder_language_labels.append(1)

                labels.append(l2_tokenization[i].ids)
            else:
                raise AssertionError
        return {
            "labels": labels,

            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_ids_offset": input_ids_offset,
            "encoder_language_labels": encoder_language_labels,

            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids_offset": decoder_input_ids_offset,
            "decoder_language_labels": decoder_language_labels,
        }

    return prepare
