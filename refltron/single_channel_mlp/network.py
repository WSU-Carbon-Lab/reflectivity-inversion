"""The single-channel MLP: the original fixed-q Keras network of code-old, in PyTorch.

OldNet takes one curve on a q grid shared by every sample (log R_s, log R_p, or an
s/p anisotropy ratio; see train.py in this folder) and predicts the film parameters
scaled to [0, 1] inside their training ranges.
"""

import torch
from torch import Tensor, nn


class OldNet(nn.Module):
    """Fixed-q network: the original Keras model of code-old, ported line for line.

    Source: ``define_model`` in code-old/training.py (Greco et al.). Checked against the
    saved code-old/trained_model.hdf5: its weights load into this class and give the
    same outputs.

    Input, for every sample in a batch:
        one curve on a FIXED grid of n_q q values. In the original this is the natural
        log of the reflectivity (``prepare_old_input``); train.py can also feed log(P)
        or an s/p anisotropy ratio. q itself is not an input: every curve shares the same
        grid, so a number's position in the vector says which q it belongs to.

    Output: one number per label, scaled to [0, 1] by that label's (min, max) range.

    How data flows through the network (hidden widths 400, 800, 400, 300, 200, 100):

        curve    (batch, n_q)
          -> Linear n_q -> 400,      ReLU
          -> Linear 400 -> 800,      ReLU
          -> Linear 800 -> 400,      ReLU
          -> Linear 400 -> 300,      ReLU
          -> Linear 300 -> 200,      ReLU
          -> Linear 200 -> 100,      ReLU
          -> Linear 100 -> n_labels, ReLU   (ReLU on the output too, as in the original)

    One deliberate difference from the original: the layers start from PyTorch's
    default nn.Linear values (small random weights and biases), not the Keras defaults
    (Glorot uniform weights, zero biases). With the Keras start on our data, the first
    few steps pushed every output below 0, and the output ReLU then froze the whole
    network (no gradient flows through a ReLU that outputs 0).

    The output ReLU itself is kept: predictions can never go below 0, which suits
    labels scaled to [0, 1].

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

    def forward(self, curve: Tensor) -> Tensor:
        """(batch, n_q) input curve -> (batch, n_labels) scaled labels."""
        return self.layers(curve)


def prepare_old_input(reflectivity: Tensor) -> Tensor:
    """OldNet's only input transform: the natural log of the reflectivity.

    The original applied ``np.log`` with no floor, so every reflectivity must be > 0.
    """
    return torch.log(reflectivity)
