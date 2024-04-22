from torch import nn

# --------------------------------------
# work around way for passing multiple inputs to nn.sequential(*layers)
# --------------------------------------
class sequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
