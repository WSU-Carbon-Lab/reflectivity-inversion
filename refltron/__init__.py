"""Neural-network inversion of polarized X-ray reflectivity into thin-film parameters.

    common/              shared by every model: physics, labels and losses, plot style, W&B settings
    dataset/             make and publish the reflectivity-1m datasets (Hugging Face)
    multi_channel_mlp/   Refltron: (s, p, q) channels in, variable-q
    single_channel_mlp/  the original fixed-q Keras MLP: one curve in

Run everything from the project folder as a module, for example:

    uv run python -m refltron.single_channel_mlp.train --input s
"""
