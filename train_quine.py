import time
import argparse
import jax
import jax.numpy as jnp
from jax import random, jit
import optax

# helpers from quine.py
from quine import (
    init_mlp_params,
    create_dataset,
    v_apply_n,
    linearize,
    weightwise_application,
    create_pca_plot,
)


# -------------------------
# Utilities
# -------------------------

def flatten_dataset(dataset):
    """Concatenate list of (v_flat, l_flat, c_flat, p_flat) tuples into batch arrays."""
    v_list = [t[0] for t in dataset]
    l_list = [t[1] for t in dataset]
    c_list = [t[2] for t in dataset]
    p_list = [t[3] for t in dataset]
    v = jnp.concatenate(v_list)
    l = jnp.concatenate(l_list)
    c = jnp.concatenate(c_list)
    p = jnp.concatenate(p_list)
    return v, l, c, p


def build_optimizer(lr):
    return optax.adam(lr)


def _check_for_invalid(params, context_msg=""):
    for layer in params:
        if jnp.isnan(layer["w"]).any() or jnp.isinf(layer["w"]).any() or jnp.isnan(layer["b"]).any() or jnp.isinf(layer["b"]).any():
            raise ValueError(f"NaN or Inf detected. {context_msg}")


# -------------------------
# Training helpers
# -------------------------

def make_loss_and_update(optimizer):
    """Return JIT-ed loss_fn and update function bound to the optimizer."""

    @jit
    def loss_fn(params, v_batch, l_batch, c_batch, p_batch, y_batch):
        preds = v_apply_n(params, v_batch, l_batch, c_batch, p_batch).squeeze(-1)
        return jnp.mean((preds - y_batch) ** 2)

    @jit
    def update(params, opt_state, v_batch, l_batch, c_batch, p_batch, y_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, v_batch, l_batch, c_batch, p_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return loss_fn, update


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3, help="Number of alternation cycles")
    parser.add_argument("--train_epochs_per_cycle", type=int, default=5)
    parser.add_argument("--apply_steps_per_cycle", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--self_train", action="store_true")
    parser.add_argument("--seed", type=int, default=int(time.time()))
    args = parser.parse_args()

    key = random.PRNGKey(args.seed)
    m_key, n_key = random.split(key)

    # initialize target network M and processor N
    m_params = init_mlp_params([3, args.hidden_size, 2], m_key)
    N_INPUT = 4
    N_OUTPUT = 1
    n_params = init_mlp_params([N_INPUT, args.hidden_size, N_OUTPUT], n_key)

    optimizer = build_optimizer(args.lr)
    opt_state = optimizer.init(n_params)

    loss_fn, update = make_loss_and_update(optimizer)

    # trajectory of processor parameters across all training epochs
    trajectory = [linearize(n_params)]

    for cycle in range(args.cycles):
        # --- training phase: fit N to predict current M params ---
        dataset = create_dataset(m_params) if not args.self_train else create_dataset(n_params)
        v, l, c, p = flatten_dataset(dataset)
        y = v

        print(f"Cycle {cycle}: training N for {args.train_epochs_per_cycle} epochs to match current target")
        for epoch in range(1, args.train_epochs_per_cycle + 1):
            n_params, opt_state, loss = update(n_params, opt_state, v, l, c, p, y)

            _check_for_invalid(n_params, context_msg=f"cycle {cycle}")

            trajectory.append(linearize(n_params))
            if epoch % max(1, args.train_epochs_per_cycle // 5) == 0 or epoch <= 2:
                print(f"  train epoch={epoch:3d} loss={float(loss):.6e}")

        # --- weightwise application phase: apply N to N a few times ---
        print(f"Cycle {cycle}: applying N to N {args.apply_steps_per_cycle} times (weightwise)")
        for i in range(args.apply_steps_per_cycle):
            n_params = weightwise_application(n_params, n_params)
            _check_for_invalid(n_params, context_msg=f"cycle {cycle} after apply")

    # PCA visualization of the processor parameter trajectory (delegated to helper)
    try:
        trajectory = jnp.stack(trajectory)
        create_pca_plot(trajectory, "trained_quine.png", title='Processor (N) parameter trajectory during alternating train/apply')
    except Exception as e:
        print('Could not create PCA plot:', e)


if __name__ == '__main__':
    main()
