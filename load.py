from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder

LABELS = ['Af_a', 'Af_d', 'Af_j', 'Af_n', 'An_a', 'An_d', 'An_j', 'An_n', 'Bf_a', 'Bf_d',
 'Bf_j', 'Bf_n', 'Bn_a', 'Bn_d', 'Bn_j', 'Bn_n', 'Cn_a', 'Cn_d', 'Cn_j', 'Cn_n',
 'Df_a', 'Df_d', 'Df_j', 'Df_n', 'Dn_a', 'Dn_d', 'Dn_j', 'Dn_n', 'Ef_a', 'Ef_d',
 'Ef_j', 'Ef_n', 'En_a', 'En_d', 'En_j', 'En_n', 'Fn_a', 'Fn_d', 'Fn_j', 'Fn_n',
 'Gf_a', 'Gf_d', 'Gf_j', 'Gf_n', 'Gn_a', 'Gn_d', 'Gn_j', 'Gn_n']

CHORD_TYPE_NAMES = {
    "a": "Augmented",
    "d": "Diminished",
    "j": "Major",
    "n": "Minor",
}

def load_feature(name: str, cache_dir: str, label_encoder = None):
    x_path = Path(cache_dir) / f"{name}_x.npy"
    y_path = Path(cache_dir) / f"{name}_y.npy"
    if x_path.exists() and y_path.exists():

        x = np.load(x_path)
        y = np.load(y_path)

        
        if name == "train":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            
            return x, y, label_encoder

        else:
            y = label_encoder.transform(y)
            return x, y
    