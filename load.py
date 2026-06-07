from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

# 12 base notes (7 natural + 5 flat) x 4 triad types = 48 classes.
# Encoding: {note}{f|n}_{type}
#   note  : A-G
#   f|n   : flat (f) or natural (n)
#   type  : a=augmented, d=diminished, j=major, n=minor
LABELS = [
    "Af_a", "Af_d", "Af_j", "Af_n",
    "An_a", "An_d", "An_j", "An_n",
    "Bf_a", "Bf_d", "Bf_j", "Bf_n",
    "Bn_a", "Bn_d", "Bn_j", "Bn_n",
    "Cn_a", "Cn_d", "Cn_j", "Cn_n",
    "Df_a", "Df_d", "Df_j", "Df_n",
    "Dn_a", "Dn_d", "Dn_j", "Dn_n",
    "Ef_a", "Ef_d", "Ef_j", "Ef_n",
    "En_a", "En_d", "En_j", "En_n",
    "Fn_a", "Fn_d", "Fn_j", "Fn_n",
    "Gf_a", "Gf_d", "Gf_j", "Gf_n",
    "Gn_a", "Gn_d", "Gn_j", "Gn_n",
]

# Human-readable chord type names keyed by the single-letter code used in
# filenames and LABELS.
CHORD_TYPE_NAMES = {
    "a": "Augmented",
    "d": "Diminished",
    "j": "Major",
    "n": "Minor",
}


def _make_label_encoder() -> LabelEncoder:
    """Return a LabelEncoder already fit on the full 48-class LABELS list.

    Fitting on the full list (rather than on whatever classes happen to appear
    in the training split) guarantees that every class always maps to the same
    integer index regardless of split composition.
    """
    le = LabelEncoder()
    le.fit(LABELS)
    return le


def load_feature(name: str, cache_dir: str, label_encoder=None):
    """Load a feature split from a cache directory.

    Parameters
    ----------
    name:
        Split name, one of "train", "val", or "test".
    cache_dir:
        Directory containing ``{name}_x.npy`` and ``{name}_y.npy``.
    label_encoder:
        Must be ``None`` for the training split (a new encoder is created and
        returned).  Must be the encoder returned from the training call for
        val/test splits.

    Returns
    -------
    For the training split: ``(x, y, label_encoder)``
    For val/test splits:    ``(x, y)``
    """
    x_path = Path(cache_dir) / f"{name}_x.npy"
    y_path = Path(cache_dir) / f"{name}_y.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing cache files for '{name}' in '{cache_dir}'. "
            f"Expected {x_path.name} and {y_path.name}."
        )

    x = np.load(x_path)
    y = np.load(y_path)

    if name == "train":
        # Always fit on the full LABELS list so that integer indices are
        # consistent across all splits and match the model output neurons.
        label_encoder = _make_label_encoder()
        y = label_encoder.transform(y)
        return x, y, label_encoder

    if label_encoder is None:
        raise ValueError(
            "label_encoder is required for validation and test splits. "
            "Pass the encoder returned by load_feature('train', ...)."
        )

    y = label_encoder.transform(y)
    return x, y
