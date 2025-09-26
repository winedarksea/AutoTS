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

tqdm = lambda x, **kwargs: x  # fake version of tqdm

class norm(object):  # fake version of scipy.stats.norm
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def ppf(self, q):
        """Simple normal quantile approximation using Abramowitz & Stegun rational approximation."""
        # Claude came up with this, so let's hope the LLM got it right.
        import math
        if q <= 0 or q >= 1:
            raise ValueError("q must be in (0, 1)")
        
        # Use symmetry: if q > 0.5, compute for 1-q and negate
        if q > 0.5:
            return -self.ppf(1 - q)
        
        # Rational approximation for q in (0, 0.5]
        t = math.sqrt(-2 * math.log(q))
        z = t - (2.515517 + 0.802853*t + 0.010328*t*t) / (1 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)
        return z * self.scale + self.loc

    def cdf(self, x):
        """Cumulative distribution function using error function."""
        import math
        return 0.5 * (1 + math.erf((x - self.loc) / (self.scale * math.sqrt(2))))
    