import jax
from jax import nn
import jax.numpy as jnp
from jax import random, vmap, jit
from copy import deepcopy


def init_mlp_params(layer_widths, key):
    """Initializes an MLP using He Normal for weights and zeros for biases."""
    params = []
    # Split the main key into enough subkeys for each layer
    keys = random.split(key, len(layer_widths) - 1)
    
    for i in range(len(layer_widths) - 1):
        n_in = layer_widths[i]
        n_out = layer_widths[i+1]
        
        # Split key for weight and bias
        w_key, b_key = random.split(keys[i])
        
        # He Initialization: std = sqrt(2 / n_in)
        stddev = jnp.sqrt(2.0 / n_in)
        
        params.append({
            'w': random.normal(w_key, (n_in, n_out)) * stddev,
            'b': jnp.zeros((n_out,))
        })
    return params


def add_noise(params, key, sigma=0.1, percentage=1):
    """Add Gaussian noise to each parameter"""
    assert 0 <= percentage <= 1, "Percentage must be between 0 and 1"
    new_params = []
    for layer in params:
        d = {'w': None, 'b': None}
        key, use_key = random.split(key)
        mask = random.bernoulli(use_key, p=percentage, shape=layer['w'].shape)
        key, use_key = random.split(key)
        noise = sigma * jax.random.normal(use_key, shape=layer['w'].shape)
        d['w'] = layer['w'] + (mask * noise)
        key, use_key = random.split(key)
        mask = random.bernoulli(use_key, p=percentage, shape=layer['b'].shape)
        noise = sigma * jax.random.normal(key, shape=layer['b'].shape)
        d['b'] = layer['b'] + (mask * noise)
        new_params.append(d)
    return new_params


def linearize(params):
    """Flattens the MLP parameters into a single vector."""
    flat_params = []
    for layer in params:
        flat_params.append(layer['w'].flatten())
        flat_params.append(layer['b'].flatten())
    return jnp.concatenate(flat_params) 


def forward(params, x):
    """Standard forward pass with tanh activation."""
    *hidden, last = params
    for layer in hidden:
        x = nn.relu(jnp.dot(x, layer['w']) + layer['b'])
    return jnp.dot(x, last['w']) + last['b']


def apply_n_to_weight(n_params, v, l, c, p):
    """
    The processor function N(v, l, c, p).
    Expects a single weight and its normalized coordinates.
    """
    # Concatenate inputs into a vector of size 4
    n_input = jnp.array([v, l, c, p])
    return forward(n_params, n_input)


# Vectorize N so it can handle batches of weights/coordinates simultaneously
v_apply_n = jit(vmap(apply_n_to_weight, in_axes=(None, 0, 0, 0, 0)))


def create_dataset(params):
    """
    Creates a dataset of (v, l, c, p) for all weights and biases.
    v: the weight/bias value
    l: normalized layer index
    c: normalized output unit index (for weights) or bias unit index (for biases)
    p: normalized input unit index (for weights) or 1.0 for biases
    """
    dataset = []
    num_layers = len(params)
    
    for l_idx, layer in enumerate(params):
        w, b = layer['w'], layer['b']
        in_dim, out_dim = w.shape
        l_norm = l_idx / (num_layers - 1) if num_layers > 1 else 0.0
        
        # --- Process Weights (W) ---
        # Generate normalized coordinates for the weight matrix
        if l_idx < num_layers - 1:
            c_coords = jnp.linspace(0, 1, out_dim+1)
            c_coords_w = c_coords[:-1]
        else:
            # last layer
            c_coords_w = jnp.linspace(0, 1, out_dim)
        p_coords = jnp.linspace(0, 1, in_dim+1)
        p_coords_w = p_coords[:-1]

        # Create grids for c (cell/output) and p (position/input)
        grid_c, grid_p = jnp.meshgrid(c_coords_w, p_coords_w, indexing='ij')
        
        # Flatten for vectorized application
        v_flat = w.T.reshape(-1) # Match grid orientation
        l_flat = jnp.full_like(v_flat, l_norm)
        c_flat = grid_c.reshape(-1)
        p_flat = grid_p.reshape(-1)

        dataset.append((v_flat, l_flat, c_flat, p_flat))
        
        # --- Process Biases (b) ---
        b_v_flat = b
        b_l_flat = jnp.full_like(b_v_flat, l_norm)
        b_c_flat = c_coords_w if l_idx < num_layers -1 else jnp.linspace(0, 1, out_dim) # Biases correspond to output units
        b_p_flat = jnp.full_like(b_v_flat, 1.0) # Bias is constant position
        
        dataset.append((b_v_flat, b_l_flat, b_c_flat, b_p_flat))
        
    return dataset


def weightwise_application(n_params, m_params):
    """
    Performs N ◁ M by mapping M's parameters through N.
    """
    new_m_params = []
    num_layers = len(m_params)
    
    for l_idx, layer in enumerate(m_params):
        w, b = layer['w'], layer['b']
        in_dim, out_dim = w.shape
        l_norm = l_idx / (num_layers - 1) if num_layers > 1 else 0.0
        
        # --- Process Weights (W) ---
        # Generate normalized coordinates for the weight matrix
        if l_idx < num_layers - 1:
            c_coords = jnp.linspace(0, 1, out_dim+1)
            c_coords_w = c_coords[:-1]
        else:
            # last layer
            c_coords_w = jnp.linspace(0, 1, out_dim)
        p_coords = jnp.linspace(0, 1, in_dim+1)
        p_coords_w = p_coords[:-1]

        # Create grids for c (cell/output) and p (position/input)
        grid_c, grid_p = jnp.meshgrid(c_coords_w, p_coords_w, indexing='ij')
        
        # Flatten for vectorized application
        v_flat = w.T.reshape(-1) # Match grid orientation
        l_flat = jnp.full_like(v_flat, l_norm)
        c_flat = grid_c.reshape(-1)
        p_flat = grid_p.reshape(-1)

        # l is the layer index, c is the unit index where the weight points to, p is the unit index where the weight comes from
        
        new_w_flat = v_apply_n(n_params, v_flat, l_flat, c_flat, p_flat)
        new_w = new_w_flat.reshape(out_dim, in_dim).T
        
        # --- Process Biases (b) ---
        b_v_flat = b
        b_l_flat = jnp.full_like(b_v_flat, l_norm)
        b_c_flat = c_coords_w if l_idx < num_layers -1 else jnp.linspace(0, 1, out_dim) # Biases correspond to output units
        b_p_flat = jnp.full_like(b_v_flat, 1.0) # Bias is constant position
        
        new_b = v_apply_n(n_params, b_v_flat, b_l_flat, b_c_flat, b_p_flat)
        new_b = new_b.squeeze(-1)
        new_m_params.append({'w': new_w, 'b': new_b})
        
    return new_m_params


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weightwise_iterations', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--self_replicate', action='store_true', help="Whether to perform self-replication (N ◁ N) or not (N ◁ M)")
    parser.add_argument('--noise_percentage', type=float, default=1.0, help="Percentage of weights/biases to which noise is applied (0 to 1)")
    parser.add_argument('--hidden_size', type=int, default=8, help="Number of hidden units in the processor network N (same in M, if present)")
    parser.add_argument('--regenerate_every', type=int, default=10, help="Regenerate the processor network every N iterations (only applies if self_replicate is True)")
    args = parser.parse_args()

    N_INPUT = 4 # value, layer index, outward node index, inward node index
    N_OUTPUT = 1 # new weight value
    
    key = random.PRNGKey(int(time.time()))
    m_key, n_key = random.split(key)

    # any custom network
    m_params = init_mlp_params([3, args.hidden_size, 2], m_key)
    key, n_key = jax.random.split(n_key)

    # processor network
    n_params = init_mlp_params([N_INPUT, args.hidden_size, N_OUTPUT], n_key)

    fast_weightwise = jit(weightwise_application)

    transformed_params = deepcopy(n_params) if args.self_replicate else deepcopy(m_params)
    
    trajectory = [linearize(transformed_params)]
    for i in tqdm(range(args.weightwise_iterations)):

        transformed_params = fast_weightwise(n_params, transformed_params)
        
        if args.self_replicate and (i+1) % args.regenerate_every == 0:
            n_params = deepcopy(transformed_params)

        detected_nan_inf = False
        for layer in transformed_params:
            if jnp.isnan(layer['w']).any() or jnp.isinf(layer['w']).any() or jnp.isnan(layer['b']).any() or jnp.isinf(layer['b']).any():
                detected_nan_inf = True
                print(f"NaN or Inf detected at iteration {i}. Stopping iterations.")
                break

        if detected_nan_inf:
            break

        key, use_key = jax.random.split(key)
        transformed_params = add_noise(transformed_params, use_key, sigma=args.sigma, 
                                         percentage=args.noise_percentage)
        trajectory.append(linearize(transformed_params))
    trajectory = jnp.array(trajectory)

    # convert trajectory from jit array to numpy for PCA
    trajectory_numpy = jax.device_get(trajectory)

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. PCA Projection (Same as before)
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(trajectory_numpy)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # 2. Prepare the transparency (Alpha) array
    # We want alpha to go from 0.1 (very transparent) to 1.0 (fully opaque)
    n_points = trajectory_2d.shape[0]
    alphas = np.linspace(0.1, 1.0, n_points)

    # 3. Create an RGBA color array
    # We'll pick a base color (e.g., Tab:Blue) and apply our alpha gradient
    base_color = plt.cm.Blues(0.8)  # Get the RGBA for a nice dark blue
    colors = np.zeros((n_points, 4))
    colors[:] = base_color         # Set RGB for all points
    colors[:, 3] = alphas          # Override the Alpha channel with our gradient

    # 4. Plotting
    plt.figure(figsize=(8, 6))

    # Plot the dots with the transparency gradient
    plt.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], c=colors, marker='o', edgecolors='none')

    # Optional: Plot a very faint line connecting them so the path is clear
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], color='gray', alpha=0.2, zorder=0)

    network_name = "N ◁ N" if args.self_replicate else "N ◁ M"
    plt.title(f'Trajectory of {network_name} (Alpha = Time)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('quine.png')
