import re
from collections import OrderedDict

import torch.nn as nn
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Module(nn.Module):
    """
    Subclass of a pytorch Module that provides API for storing intermediate
    tensors by name, which can be used to retrieve the tensor for debugging
    and visualization later.

    To store some intermediate tensor (e.g. attention weights)
    simply call self.expose(tensor, name) with the intermediate tensor
    and a name for it. The name will be local, in the sense that it will
    be appended to the module name, to form a full name
    (e.g. if you call self.expose(x, "attn_weights") and self has name
    "model.transformer.h0.cross_attention" then the full name of the tensor
    will be "model.transformer.h0.cross_attention.attn_weights" and this
    will be the name returned when you call self.named_exposed_tensors().

    You need to MANUALLY release the memory for intermediate tensors when
    you are done using them, or else there will be memory leaks.
    """

    def __init__(self):
        super().__init__()
        self.exposed = OrderedDict()    # storage for intermediate tensors
        self.exposure_patterns = set()  # set of regular expressions

    def expose(self, tensor, name):
        """
        `tensor` will be stored under `name`, if the name is matched
        by one of a set of pre-specified regex.
        """
        if any(re.match(pattern, name) for pattern in self.exposure_patterns):
            if name in self.exposed:
                logger.warning(f"Overwriting exposed tensor {name}, did you release exposed tensors after the last forward call?")
            self.exposed[name] = tensor
        return tensor

    def release_exposed_tensors(self):
        self.apply(self._release_exposed_tensors)

    def _release_exposed_tensors(self, module):
        if isinstance(module, Module):
            module.exposed = OrderedDict()

    def named_exposed_tensors(self):
        """
        Generator for iterating over all the stored tensors and their names.
        """
        for prefix, module in self.named_modules():
            if isinstance(module, Module):
                for name, tensor in module.exposed.items():
                    if prefix:
                        yield f"{prefix}.{name}", tensor
                    else:
                        yield f"{name}", tensor

    def add_exposure_pattern(self, name_pattern=".*", module_pattern=".*"):
        """
        `name_pattern` is used to match locally against the `name` argument
        supplied to self.expose. `module_pattern` is matched against module names,
        so that only selected modules will expose certain names.
        """
        any = False
        for module_name, module in self.named_modules():
            if re.match(module_pattern, module_name):
                if isinstance(module, Module):
                    module.exposure_patterns.add(name_pattern)
                    any = True
        if not any:
            raise ValueError(
                f"can only add exposure to cslm.modeling.module.Module, found none matching {module_pattern})")

    def remove_exposure_pattern(self, module_pattern=".*", name_pattern=".*"):
        """
        Same as add_exposure_pattern but just removes it instead of adding.
        This should rarely be necessary.
        """
        any = False
        for module_name, module in self.named_modules():
            if re.match(module_pattern, module_name):
                if isinstance(module, Module):
                    module.exposure_patterns.remove(name_pattern)
                    any = True
        if not any:
            raise ValueError(
                f"can only remove exposure from cslm.modeling.module.Module, found none matching {module_pattern})")
