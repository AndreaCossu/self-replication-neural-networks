import time
import jax
from jax import nn, random, vmap, jit
import jax.numpy as jnp
from copy import deepcopy

# -------------------------
# MLP / Parameter utilities
# -------------------------

def init_mlp_params(layer_widths, key):
    """Initialize an MLP using He normal for weights and zeros for biases.

    Returns a list of dicts: [{'w': ..., 'b': ...}, ...]
    """
    params = []
    keys = random.split(key, len(layer_widths) - 1)

    for i in range(len(layer_widths) - 1):
        n_in = layer_widths[i]
        n_out = layer_widths[i + 1]
        w_key, b_key = random.split(keys[i])
        stddev = jnp.sqrt(2.0 / n_in)
        params.append({
            "w": random.normal(w_key, (n_in, n_out)) * stddev,
            "b": jnp.zeros((n_out,)),
        })
    return params


def linearize(params):
    """Flatten parameters to a single 1D vector."""
    flat = []
    for layer in params:
        flat.append(layer["w"].ravel())
        flat.append(layer["b"].ravel())
    return jnp.concatenate(flat)


def add_noise(params, key, sigma=0.1, percentage=1.0):
    """Add Gaussian noise to parameters; `percentage` is fraction of entries affected."""
    assert 0.0 <= percentage <= 1.0
    if percentage == 0.0:
        return params  # No noise to add
    
    new_params = []
    for layer in params:
        d = {}
        key, use_key = random.split(key)
        mask_w = random.bernoulli(use_key, p=percentage, shape=layer["w"].shape)
        key, use_key = random.split(key)
        noise_w = sigma * random.normal(use_key, shape=layer["w"].shape)
        d["w"] = layer["w"] + mask_w * noise_w
        key, use_key = random.split(key)
        mask_b = random.bernoulli(use_key, p=percentage, shape=layer["b"].shape)
        key, use_key = random.split(key)
        noise_b = sigma * random.normal(use_key, shape=layer["b"].shape)
        d["b"] = layer["b"] + mask_b * noise_b
        new_params.append(d)
    return new_params


# -------------------------
# Forward / processor N
# -------------------------

def forward(params, x):
    """Simple MLP forward with ReLU hidden activations."""
    *hidden, last = params
    for layer in hidden:
        x = nn.relu(jnp.dot(x, layer["w"]) + layer["b"])
    return jnp.dot(x, last["w"]) + last["b"]


def apply_n_to_weight(n_params, v, l, c, p):
    """Processor N applied to a single scalar weight with normalized coords.

    Inputs: v, l, c, p are scalars; output is a 1-element array.
    """
    n_input = jnp.array([v, l, c, p])
    return forward(n_params, n_input)


# Vectorize and JIT the processor for batch application
v_apply_n = jit(vmap(apply_n_to_weight, in_axes=(None, 0, 0, 0, 0)))


# -------------------------
# Coordinate helpers
# -------------------------

def _normalized_layer_index(l_idx, num_layers):
    return l_idx / (num_layers - 1) if num_layers > 1 else 0.0


def _coords_for_layer(w_shape, l_idx, num_layers, is_last_layer):
    """Return flattened grids and coordinate arrays for a given layer's weights.

    Returns (l_norm, c_flat, p_flat, out_dim, in_dim, c_coords_w)
    """
    in_dim, out_dim = w_shape
    l_norm = _normalized_layer_index(l_idx, num_layers)

    # c coordinates (output units)
    if not is_last_layer:
        c_coords = jnp.linspace(0.0, 1.0, out_dim + 1)
        c_coords_w = c_coords[:-1]
    else:
        c_coords_w = jnp.linspace(0.0, 1.0, out_dim)

    # p coordinates (input units)
    p_coords = jnp.linspace(0.0, 1.0, in_dim + 1)
    p_coords_w = p_coords[:-1]

    grid_c, grid_p = jnp.meshgrid(c_coords_w, p_coords_w, indexing="ij")
    return l_norm, grid_c.reshape(-1), grid_p.reshape(-1), out_dim, in_dim, c_coords_w


# -------------------------
# Dataset creation
# -------------------------

def create_dataset(params):
    """Create dataset of tuples (v_flat, l_flat, c_flat, p_flat) for each layer.

    Each tuple corresponds to either all weights of a layer or biases of a layer.
    """
    dataset = []
    num_layers = len(params)

    for l_idx, layer in enumerate(params):
        w = layer["w"]
        b = layer["b"]
        is_last = (l_idx == num_layers - 1)

        # weights
        l_norm, c_flat, p_flat, out_dim, in_dim, c_coords_w = _coords_for_layer(w.shape, l_idx, num_layers, is_last)
        v_flat = w.T.reshape(-1)
        l_flat = jnp.full_like(v_flat, l_norm)
        dataset.append((v_flat, l_flat, c_flat, p_flat))

        # biases
        b_v_flat = b
        b_l_flat = jnp.full_like(b_v_flat, l_norm)
        b_c_flat = c_coords_w if not is_last else jnp.linspace(0.0, 1.0, out_dim)
        b_p_flat = jnp.full_like(b_v_flat, 1.0)
        dataset.append((b_v_flat, b_l_flat, b_c_flat, b_p_flat))

    return dataset


# -------------------------
# Weightwise application
# -------------------------

def _apply_n_to_layer(n_params, layer, l_idx, num_layers):
    """Apply processor N across weights and biases of a single layer and return new layer."""
    w = layer["w"]
    b = layer["b"]
    is_last = (l_idx == num_layers - 1)

    # weights
    l_norm, c_flat, p_flat, out_dim, in_dim, c_coords_w = _coords_for_layer(w.shape, l_idx, num_layers, is_last)
    v_flat = w.T.reshape(-1)
    new_w_flat = v_apply_n(n_params, v_flat, jnp.full_like(v_flat, l_norm), c_flat, p_flat)
    new_w = new_w_flat.reshape(out_dim, in_dim).T

    # biases
    b_v_flat = b
    b_l_flat = jnp.full_like(b_v_flat, l_norm)
    b_c_flat = c_coords_w if not is_last else jnp.linspace(0.0, 1.0, out_dim)
    b_p_flat = jnp.full_like(b_v_flat, 1.0)
    new_b = v_apply_n(n_params, b_v_flat, b_l_flat, b_c_flat, b_p_flat).squeeze(-1)

    return {"w": new_w, "b": new_b}


def weightwise_application(n_params, m_params):
    """Apply processor N elementwise to all weights/biases of M and return transformed M params."""
    new_params = []
    num_layers = len(m_params)
    for l_idx, layer in enumerate(m_params):
        new_layer = _apply_n_to_layer(n_params, layer, l_idx, num_layers)
        new_params.append(new_layer)
    return new_params


# -------------------------
# PCA / Plot helper
# -------------------------

def create_pca_plot(trajectory, outpath, title=None, cmap_name="Blues"):
    """Compute PCA on a trajectory (jax or numpy array) and save a 2D scatter plot.

    trajectory: array-like of shape (n_points, dim)
    outpath: path to save PNG
    title: optional plot title
    cmap_name: matplotlib colormap name
    """
    # local imports so these heavy deps are optional when functions are used programmatically
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    import jax

    trajectory = jax.device_get(trajectory)
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(trajectory)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    n_points = trajectory_2d.shape[0]
    alphas = np.linspace(0.1, 1.0, n_points)
    cmap = plt.get_cmap(cmap_name)
    base_color = cmap(0.8)
    colors = np.zeros((n_points, 4))
    colors[:] = base_color
    colors[:, 3] = alphas

    plt.figure(figsize=(8, 6))
    plt.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], c=colors, marker="o", edgecolors="none")
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], color="gray", alpha=0.2, zorder=0)
    if title:
        plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(outpath)
    print(f"Saved PCA plot to {outpath}")


# -------------------------
# If run as script (retain original demo behavior)
# -------------------------

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--weightwise_iterations", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--self_replicate", action="store_true")
    parser.add_argument("--noise_percentage", type=float, default=0)
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--regenerate_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=int(time.time()))
    args = parser.parse_args()

    N_INPUT = 4
    N_OUTPUT = 1

    key = random.PRNGKey(args.seed)
    m_key, n_key = random.split(key)

    m_params = init_mlp_params([3, args.hidden_size, 2], m_key)
    key, n_key = random.split(n_key)
    n_params = init_mlp_params([N_INPUT, args.hidden_size, N_OUTPUT], n_key)

    fast_weightwise = jit(weightwise_application)

    transformed_params = deepcopy(n_params) if args.self_replicate else deepcopy(m_params)

    trajectory = [linearize(transformed_params)]
    for i in tqdm(range(args.weightwise_iterations)):
        transformed_params = fast_weightwise(n_params, transformed_params)

        if args.self_replicate and (i + 1) % args.regenerate_every == 0:
            n_params = deepcopy(transformed_params)

        # NaN/Inf guard
        detected_nan_inf = False
        for layer in transformed_params:
            if jnp.isnan(layer["w"]).any() or jnp.isinf(layer["w"]).any() or jnp.isnan(layer["b"]).any() or jnp.isinf(layer["b"]).any():
                detected_nan_inf = True
                print(f"NaN or Inf detected at iteration {i}. Stopping iterations.")
                break
        if detected_nan_inf:
            break

        key, use_key = random.split(key)
        transformed_params = add_noise(transformed_params, use_key, sigma=args.sigma, percentage=args.noise_percentage)
        trajectory.append(linearize(transformed_params))

    trajectory = jnp.array(trajectory)

    network_name = "N ◁ N" if args.self_replicate else "N ◁ M"
    create_pca_plot(trajectory, "quine.png", title=f"Trajectory of {network_name} (Alpha = Time)")
