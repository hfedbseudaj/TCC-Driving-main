"""Save and load complete supervised-learning bargaining checkpoints."""

import os

import torch


SL_CHECKPOINT_SCHEMA_VERSION = 3
SL_CHECKPOINT_COMPONENTS = (
    "state_encoder",
    "offer_decoder",
    "utility_estimator",
    "offer_to_waypoint",
)


def read_sl_checkpoint(checkpoint_path, map_location="cpu"):
    """Read a checkpoint created by this project's trusted SL training code.

    PyTorch 2.6 changed torch.load's default to weights_only=True. SL metadata
    may contain NumPy scalars, so explicitly opt into the legacy loader for
    these locally produced checkpoints while retaining compatibility with
    older PyTorch releases that do not accept weights_only.
    """
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def build_sl_checkpoint(model, optimizer=None, **metadata):
    """Build a checkpoint containing the exact full inference model state."""
    checkpoint = {
        "checkpoint_schema_version": SL_CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
    }
    for name in SL_CHECKPOINT_COMPONENTS:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"SL model is missing checkpoint component: {name}")
        checkpoint[name] = module.state_dict()
    if optimizer is not None:
        checkpoint["opt"] = optimizer.state_dict()
    checkpoint.update(metadata)
    return checkpoint


def load_sl_checkpoint(model, checkpoint_or_path, map_location="cpu", load_utility=True):
    """Load a complete SL checkpoint; reject files missing any inference component."""
    if isinstance(checkpoint_or_path, (str, bytes, os.PathLike)):
        checkpoint = read_sl_checkpoint(checkpoint_or_path, map_location=map_location)
    else:
        checkpoint = checkpoint_or_path

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid SL checkpoint: expected a dictionary of model components.")

    if "model_state_dict" in checkpoint and load_utility:
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError("Incompatible full SL model state: " + str(exc))
        return checkpoint

    required = tuple(
        name for name in SL_CHECKPOINT_COMPONENTS
        if load_utility or name != "utility_estimator"
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(
            "Incomplete legacy SL checkpoint: missing component(s) "
            + ", ".join(missing)
            + ". It cannot represent the complete trained model; retrain with the current checkpoint format."
        )

    for name in required:
        if name == "utility_estimator" and not load_utility:
            continue
        module = getattr(model, name, None)
        if module is None:
            raise ValueError("Cannot load SL checkpoint: model has no component " + name + ".")
        try:
            module.load_state_dict(checkpoint[name], strict=True)
        except RuntimeError as exc:
            raise ValueError("Incompatible SL checkpoint component " + name + ": " + str(exc))

    return checkpoint


def load_sl_checkpoint_as_policy(model, policy, checkpoint_or_path, map_location="cpu"):
    """Initialize an RL model and policy from one complete SL checkpoint.

    The state encoder, utility estimator and offer-to-waypoint transform are
    loaded into the bargaining model. The supervised offer decoder initializes
    the stochastic policy's mean network; its exploration scale remains under
    the RL algorithm's control.
    """
    checkpoint = load_sl_checkpoint(
        model, checkpoint_or_path, map_location=map_location, load_utility=True
    )
    decoder_state = checkpoint.get("offer_decoder")
    if decoder_state is None:
        raise ValueError("SL checkpoint is missing the offer_decoder component.")
    mean_state = {
        (key[len("decoder."):] if key.startswith("decoder.") else key): value
        for key, value in decoder_state.items()
    }
    try:
        policy.mean_net.load_state_dict(mean_state, strict=True)
    except RuntimeError as exc:
        raise ValueError("SL offer decoder is incompatible with the RL policy: " + str(exc))
    return checkpoint
