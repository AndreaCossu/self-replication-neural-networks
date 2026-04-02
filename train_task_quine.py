import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import optax
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# helpers from quine.py
from quine import (
    init_mlp_params,
    create_dataset,
    linearize,
    _coords_for_layer,
    create_pca_plot,
    forward,
)


def flatten_dataset(dataset):
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
    # params is a pytree containing single shared network under 'net'
    for layer in params:
        if jnp.isnan(layer["w"]).any() or jnp.isinf(layer["w"]).any() or jnp.isnan(layer["b"]).any() or jnp.isinf(layer["b"]).any():
            raise ValueError(f"NaN or Inf detected in shared network. {context_msg}")


def make_loss_and_update(optimizer, task_weight=1.0):
    """Return loss component function and JIT-ed update function.

    Loss = MSE_on_quine + task_weight * cross_entropy_on_MNIST
    """

    def loss_components(params, v_batch, l_batch, c_batch, p_batch, y_batch, imgs, labels):
        # replication prediction: first output unit
        outputs = v_apply_shared(params, v_batch, l_batch, c_batch, p_batch)
        preds = outputs[:, 0]
        mse = jnp.mean((preds - y_batch) ** 2)

        # classifier logits and cross-entropy: outputs second..end
        imgs_j = jnp.array(imgs)
        batch_size = imgs_j.shape[0]
        zeros4 = jnp.zeros((batch_size, 4))
        inputs = jnp.concatenate([zeros4, imgs_j], axis=1)
        logits = vmap(lambda x: forward(params, x))(inputs)[:, 1:]
        ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        return mse, ce

    def loss_fn(params, v_batch, l_batch, c_batch, p_batch, y_batch, imgs, labels):
        mse, ce = loss_components(params, v_batch, l_batch, c_batch, p_batch, y_batch, imgs, labels)
        return mse + task_weight * ce

    @jit
    def update(params, opt_state, v_batch, l_batch, c_batch, p_batch, y_batch, imgs, labels):
        loss, grads = jax.value_and_grad(loss_fn)(params, v_batch, l_batch, c_batch, p_batch, y_batch, imgs, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return loss_components, update


def prepare_mnist(batch_size, train=True):
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def gen():
        for imgs, labels in dataloader:
            imgs_np = imgs.view(imgs.size(0), -1).numpy()
            labels_np = labels.numpy()
            yield imgs_np, labels_np

    return gen()


def v_apply_shared(net_params, v_batch, l_batch, c_batch, p_batch, img_dim=28*28):
    """Apply shared network to replication inputs; returns outputs array shape (batch, out_dim)."""
    base = jnp.stack([v_batch, l_batch, c_batch, p_batch], axis=1)
    batch_size = base.shape[0]
    zeros_img = jnp.zeros((batch_size, img_dim))
    inputs = jnp.concatenate([base, zeros_img], axis=1)
    return vmap(lambda x: forward(net_params, x))(inputs)


def _apply_shared_to_layer(net_params, layer, l_idx, num_layers, img_dim=28*28):
    w = layer["w"]
    b = layer["b"]
    is_last = (l_idx == num_layers - 1)

    # weights
    l_norm, c_flat, p_flat, out_dim, in_dim, c_coords_w = _coords_for_layer(w.shape, l_idx, num_layers, is_last)
    v_flat = w.T.reshape(-1)
    base = jnp.stack([v_flat, jnp.full_like(v_flat, l_norm), c_flat, p_flat], axis=1)
    zeros_img = jnp.zeros((base.shape[0], img_dim))
    inputs = jnp.concatenate([base, zeros_img], axis=1)
    new_w_flat = vmap(lambda x: forward(net_params, x))(inputs)[:, 0]
    new_w = new_w_flat.reshape(out_dim, in_dim).T

    # biases
    b_v_flat = b
    b_l_flat = jnp.full_like(b_v_flat, l_norm)
    b_c_flat = c_coords_w if not is_last else jnp.linspace(0.0, 1.0, out_dim)
    b_p_flat = jnp.full_like(b_v_flat, 1.0)
    b_base = jnp.stack([b_v_flat, b_l_flat, b_c_flat, b_p_flat], axis=1)
    zeros_b = jnp.zeros((b_base.shape[0], img_dim))
    b_inputs = jnp.concatenate([b_base, zeros_b], axis=1)
    new_b = vmap(lambda x: forward(net_params, x))(b_inputs)[:, 0]

    return {"w": new_w, "b": new_b}


def local_weightwise_application(net_params, m_params):
    new_params = []
    num_layers = len(m_params)
    for l_idx, layer in enumerate(m_params):
        new_layer = _apply_shared_to_layer(net_params, layer, l_idx, num_layers)
        new_params.append(new_layer)
    return new_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3, help="Number of alternation cycles")
    parser.add_argument("--train_epochs_per_cycle", type=int, default=5)
    parser.add_argument("--apply_steps_per_cycle", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=int(time.time()))
    parser.add_argument("--task_weight", type=float, default=1.0, help="Weight for MNIST cross-entropy term")
    parser.add_argument("--mnist_batch", type=int, default=128)
    args = parser.parse_args()

    key = random.PRNGKey(args.seed)
    net_key = key

    # shared network input/output sizes
    IMG_DIM = 28 * 28
    CLF_OUTPUT = 10
    N_INPUT = 4 + IMG_DIM
    N_OUTPUT = 1 + CLF_OUTPUT

    params = init_mlp_params([N_INPUT, args.hidden_size, N_OUTPUT], net_key)


    optimizer = build_optimizer(args.lr)
    opt_state = optimizer.init(params)

    loss_components, update = make_loss_and_update(optimizer, task_weight=args.task_weight)

    # trajectory of processor parameters across all training epochs
    trajectory = [linearize(params)]

    # prepare MNIST iterator
    mnist_ds = prepare_mnist(args.mnist_batch)
    mnist_iter = iter(mnist_ds)

    for cycle in range(args.cycles):
        # --- training phase: fit N (and classifier) ---
        # dataset constructed from processor parameters (no separate target M)
        dataset = create_dataset(params)
        v, l, c, p = flatten_dataset(dataset)
        y = v

        print(f"Cycle {cycle}: training N and classifier for {args.train_epochs_per_cycle} epochs")
        for epoch in range(1, args.train_epochs_per_cycle + 1):
            try:
                batch = next(mnist_iter)
            except StopIteration:
                mnist_iter = iter(prepare_mnist(args.mnist_batch))
                batch = next(mnist_iter)

            imgs_np, labels_np = batch
            imgs_np = imgs_np.astype('float32')
            imgs_np = imgs_np.reshape(imgs_np.shape[0], -1)
            labels_np = labels_np.astype('int32')

            imgs_j = jnp.array(imgs_np)
            labels_j = jnp.array(labels_np)

            params, opt_state, loss = update(params, opt_state, v, l, c, p, y, imgs_j, labels_j)

            _check_for_invalid(params, context_msg=f"cycle {cycle}")

            # record trajectory for shared net
            trajectory.append(linearize(params))
            if epoch % max(1, args.train_epochs_per_cycle // 5) == 0 or epoch <= 2:
                mse, ce = loss_components(params, v, l, c, p, y, imgs_np, labels_np)
                print(f"  train epoch={epoch:3d} loss_total={float(loss):.6e} mse={float(mse):.6e} ce={float(ce):.6e}")

        # --- weightwise application phase: apply N to N a few times ---
        print(f"Cycle {cycle}: applying N to N {args.apply_steps_per_cycle} times (weightwise)")
        for i in range(args.apply_steps_per_cycle):
            params = local_weightwise_application(params, params)
            _check_for_invalid(params, context_msg=f"cycle {cycle} after apply")

            trajectory.append(linearize(params))

    dataset_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    dataloader_test = DataLoader(dataset_test, batch_size=512, shuffle=False, drop_last=False)
    # evaluate accuracy on test set
    correct = 0
    total = 0
    for imgs, labels in dataloader_test:
        imgs_np = imgs.view(imgs.size(0), -1).numpy()
        labels_np = labels.numpy()
        imgs_j = jnp.array(imgs_np)
        logits = vmap(lambda x: forward(params, x))(jnp.concatenate([jnp.zeros((imgs_j.shape[0], 4)), imgs_j], axis=1))[:, 1:]
        preds = jnp.argmax(logits, axis=1)
        correct += jnp.sum(preds == labels_np)
        total += imgs_j.shape[0]
    accuracy = correct / total
    print(f"Test set accuracy: {accuracy:.4f}")


    # PCA visualization of the processor parameter trajectory (delegated to helper)
    try:
        trajectory = jnp.stack(trajectory)
        create_pca_plot(trajectory, "trained_task_quine.png", title='Processor (N) parameter trajectory during alternating train/apply (task-aware)')
    except Exception as e:
        print('Could not create PCA plot:', e)


if __name__ == '__main__':
    main()
