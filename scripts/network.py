"""The neural networks: from measured reflectivity curves to film parameters.

    ReflectivityNet  "dynamic-q" (published as Refltron). Inputs are s, p and q, so every
                     curve may have its own q grid.
    OldNet           "fixed-q". The original Keras network from code-old/training.py,
                     ported to PyTorch. Input is one log-reflectivity curve on a q grid
                     shared by every sample.

Both predict the film parameters scaled to [0, 1] inside their training ranges
(see ``loss.LABEL_NAMES``). ``loss.py`` converts them back to physical units.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# Ways to transform the raw reflectivity before ReflectivityNet sees it (see ``regularize``).
REGULARIZER_CHOICES = ("log", "rq4", "log_rq4")

# Activation functions: the non-linearity applied after a layer. Without one, a stack of
# linear layers is no more powerful than a single linear layer.
#   relu: max(0, x)                              (simple, but units can get stuck at 0)
#   gelu: x * Phi(x), a smooth version of ReLU   (Phi = standard normal CDF)
ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}
ACTIVATION_CHOICES = tuple(ACTIVATIONS)

# Reflectivity can be exactly 0 in a simulation; log10(0) is -infinity, so we clamp to this first.
LOG_FLOOR = 1e-30


class ReflectivityNet(nn.Module):
    """Dynamic-q network: (s, p, q) curves in, film parameters out.

    Input, for every sample in a batch:
        s, p  polarized reflectivity curves, n_q points each
        q     the momentum-transfer grid those points were measured at

    Because q is an input (not fixed), every curve may have its own q grid, which is
    why this is called the "dynamic-q" perceptron (published as Refltron).

    How data flows through the network:

        (s, p, q) -> regularize    -> (batch, 3, n_q)       make the curves easier to learn from
                  -> standardize   -> (batch, 3, n_q)       zero mean, unit spread per channel
                  -> channel mixer -> (batch, 8, n_q)       mix (S, P, q) at each q point
                  -> flatten       -> (batch, 8 * n_q)      one long vector per sample
                  -> MLP           -> (batch, n_labels)     dense layers down to the labels

    Hidden layer widths of the MLP: 400, 800, 400, 300, 200, 100.

    Args:
        n_q: number of points per curve.
        n_labels: number of film parameters to predict.
        mix_channels: how many mixed channels the 1x1 convolution makes from (S, P, q).
        regularizer: input transform for s and p, one of ``REGULARIZER_CHOICES``.
        activation: ``"relu"`` or ``"gelu"``.
        normalize_inputs: if True, standardize each input channel using statistics
            from the training data (call ``fit_input_normalization`` once before training).
    """

    def __init__(
        self,
        n_q: int = 256,
        n_labels: int = 7,
        mix_channels: int = 8,
        regularizer: str = "log",
        activation: str = "relu",
        normalize_inputs: bool = False,
    ) -> None:
        super().__init__()
        if regularizer not in REGULARIZER_CHOICES:
            raise ValueError(f"Unknown regularizer {regularizer!r}; choose from {REGULARIZER_CHOICES}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from {ACTIVATION_CHOICES}")
        self.regularizer = regularizer
        self.activation = activation
        self.normalize_inputs = normalize_inputs
        Activation = ACTIVATIONS[activation]

        # --- Input standardization -------------------------------------------------
        # After the log transform, S and P sit around -4 while q sits around 0.2. Inputs
        # on such different scales are hard to learn from, so we shift and scale each
        # channel to mean 0, standard deviation 1. The mean and std are "buffers":
        # saved with the weights but never changed by the optimizer.
        if normalize_inputs:
            self.register_buffer("input_mean", torch.zeros(3, 1))
            self.register_buffer("input_std", torch.ones(3, 1))

        # --- Channel mixer -------------------------------------------------------------
        # A Conv1d with kernel_size=1 looks at one q point at a time and makes
        # ``mix_channels`` weighted combinations of (S, P, q) there, with the same
        # weights at every q point. Flatten then joins all channels and points into one
        # vector: (batch, 8, 256) -> (batch, 2048).
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(3, mix_channels, kernel_size=1),
            Activation(),
            nn.Flatten(start_dim=1),
        )

        # --- Multi-layer perceptron (MLP) ----------------------------------------------
        # Each nn.Linear(a, b) is a fully connected layer: every one of its b outputs is
        # a weighted sum of all a inputs plus a bias.
        #
        # Note: the first two Linear layers have no activation between them. Two linear
        # maps in a row act like a single one, so the 800-wide layer can only carry
        # 400 independent numbers. That is the original design; it is kept so saved
        # checkpoints still load (see analysis/diagnostics.py for its effect).
        self.mlp = nn.Sequential(
            nn.Linear(n_q * mix_channels, 400),
            nn.Linear(400, 800),
            Activation(),
            nn.Linear(800, 400),
            Activation(),
            nn.Linear(400, 300),
            Activation(),
            nn.Linear(300, 200),
            Activation(),
            nn.Linear(200, 100),
            Activation(),
            nn.Linear(100, n_labels),  # no activation: outputs are not clipped to [0, 1]
        )

        # Start the output biases at 0.5, the middle of the normalized label range
        # (about 410 Å film thickness). Starting from 0 would make every first guess a
        # 20 Å film, and the depth-profile loss then learns slowly.
        nn.init.constant_(self.mlp[-1].bias, 0.5)

    def regularize(self, s: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Transform s and p, then stack (S, P, q) into shape (batch, 3, n_q).

        Reflectivity falls by ~8 orders of magnitude across a curve, so raw values
        are a poor input. The options are:
            log      S = log10(s)              compresses the range
            rq4      S = s * q^4               removes the ~1/q^4 Fresnel decay
            log_rq4  S = log10(s * q^4)        both
        """
        if self.regularizer == "log":
            S = torch.log10(s.clamp_min(LOG_FLOOR))
            P = torch.log10(p.clamp_min(LOG_FLOOR))
        elif self.regularizer == "rq4":
            q4 = q.pow(4)
            S = s * q4
            P = p * q4
        else:  # "log_rq4"
            q4 = q.pow(4)
            S = torch.log10((s * q4).clamp_min(LOG_FLOOR))
            P = torch.log10((p * q4).clamp_min(LOG_FLOOR))
        return torch.stack((S, P, q), dim=1)

    def prepare_inputs(self, s: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Regularize and (optionally) standardize: the exact input the mixer sees."""
        x = self.regularize(s, p, q)
        if self.normalize_inputs:
            x = (x - self.input_mean) / self.input_std
        return x

    @torch.no_grad()
    def fit_input_normalization(self, s: Tensor, p: Tensor, q: Tensor) -> None:
        """Measure each input channel's mean and std on training curves and store them.

        Uses training data only, so nothing about the validation set leaks into the model.
        """
        if not self.normalize_inputs:
            return
        x = self.regularize(s, p, q).double()  # float64 for an accurate mean over many rows
        mean = x.mean(dim=(0, 2))  # average over samples and q points -> one value per channel
        std = x.std(dim=(0, 2)).clamp_min(1e-12)  # never divide by zero
        self.input_mean.copy_(mean.float().unsqueeze(-1))
        self.input_std.copy_(std.float().unsqueeze(-1))

    def forward(self, s: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Predict normalized film parameters, shape (batch, n_labels)."""
        x = self.prepare_inputs(s, p, q)
        mixed = self.channel_mixer(x)
        return self.mlp(mixed)


class OldNet(nn.Module):
    """Fixed-q network: the original Keras model of code-old, ported line for line.

    Source: ``define_model`` in code-old/training.py (Greco et al.). Checked against the
    saved code-old/trained_model.hdf5: its weights load into this class and give the
    same outputs.

    Input, for every sample in a batch:
        one reflectivity curve on a FIXED grid of n_q q values, passed through the
        natural log first (``prepare_old_input``). q itself is not an input: every curve
        shares the same grid, so a number's position in the vector says which q it
        belongs to.

    Output: one number per label, scaled to [0, 1] by that label's (min, max) range.

    How data flows through the network (hidden widths 400, 800, 400, 300, 200, 100):

        log(R)   (batch, n_q)
          -> Linear n_q -> 400,      ReLU
          -> Linear 400 -> 800,      ReLU
          -> Linear 800 -> 400,      ReLU
          -> Linear 400 -> 300,      ReLU
          -> Linear 300 -> 200,      ReLU
          -> Linear 200 -> 100,      ReLU
          -> Linear 100 -> n_labels, ReLU   (ReLU on the output too, as in the original)

    Two details copied from the original so the port behaves the same:
      - Every layer starts from the Keras defaults: Glorot (Xavier) uniform weights and
        zero biases (see ``keras_linear``). PyTorch's own default for nn.Linear differs.
      - The output ReLU means predictions can never go below 0. That suits labels
        scaled to [0, 1], but an output stuck at 0 gets no gradient and stops learning.

    How the original was trained (for the training script):
        loss        mean squared error on the [0, 1]-scaled labels
        optimizer   Adam, lr 5e-4, betas (0.9, 0.999), eps 1e-8, no weight decay
        batches     128 rows, shuffled every epoch
        epochs      60
        validation  the LAST 20% of the rows (Keras' validation_split does not shuffle)
        checkpoint  keep the weights with the lowest validation loss

    Args:
        n_q: number of q points per curve (52 in the original, 256 in the fixed-q dataset).
        n_labels: number of labels to predict (4 in the original).
    """

    def __init__(self, n_q: int = 256, n_labels: int = 7) -> None:
        super().__init__()
        self.n_q = n_q
        self.n_labels = n_labels
        self.layers = nn.Sequential(
            nn.Linear(n_q, 400),
            nn.ReLU(),
            nn.Linear(400, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, n_labels),
            nn.ReLU(),
        )

    def forward(self, log_reflectivity: Tensor) -> Tensor:
        """(batch, n_q) natural-log reflectivity -> (batch, n_labels) scaled labels."""
        return self.layers(log_reflectivity)



def prepare_old_input(reflectivity: Tensor) -> Tensor:
    """OldNet's only input transform: the natural log of the reflectivity.

    The original applied ``np.log`` with no floor, so every reflectivity must be > 0.
    """
    return torch.log(reflectivity)
