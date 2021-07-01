from cslm.modeling.module import Module


class HeadBuilder:

    def __init__(self):
        self.base_model = None
        self.heads = []
        self.names = []

    def set_base_model(self, base_model):
        self.base_model = base_model

    def add_head(self, head, name):
        self.heads.append(head)
        self.names.append(name)

    def build(self):
        return Head(self.base_model, self.heads, self.names)


class Head(Module):

    def __init__(self, base_model, heads, names):
        super().__init__()
        self.base_model = base_model
        for name, head in zip(names, heads):
            self.add_module(name, head)


class LMHead(Module):
    """
    Interface for LMHead.
    """
    def forward(self,
                hidden_states=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        raise NotImplementedError