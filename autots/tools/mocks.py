"""Fake imports for when libraries aren't available, to prevent the whole package from failing to load."""

class Module:  # fake version of torch.nn.Module
    def __init__(self, *args, **kwargs):
        # Store args for debugging if needed
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        # Do nothing, return input or None
        return args[0] if args else None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        return

    def train(self, mode=True):
        return self

    def eval(self):
        return self