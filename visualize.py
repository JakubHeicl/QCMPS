from pathlib import Path
import argparse

import numpy as np
from scipy.optimize import minimize

from qcmps.loader import build_from_fcidump, hf_occupancy
from qcmps.qcmps_ansatz import prepare_blocks, evaluate_hamiltonian
from qcmps.circuit_blocks import ACUBlock, APCBlock, AUBlock, LPCBlock, LUBlock


BLOCK_TYPES = {
    "AU": AUBlock,
    "LU": LUBlock,
    "LPC": LPCBlock,
    "APC": APCBlock,
    "ACU": ACUBlock,
}


def make_energy_function(blocks, params, hamiltonian, nuclear_repulsion):
    param_list = list(params)
    n_params = len(param_list)

    def energy(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (n_params,):
            raise ValueError(f"Expected parameter vector of shape ({n_params},), got {x.shape}.")

        bind_map = dict(zip(param_list, x, strict=True))
        bound_blocks = [
            block.assign_parameters(
                {p: bind_map[p] for p in block.parameters},
                inplace=False,
            )
            for block in blocks
        ]
        return float(np.real_if_close(evaluate_hamiltonian(bound_blocks, hamiltonian, verbose=False))) + nuclear_repulsion

    return energy


def random_unit_vector(rng: np.random.Generator, n_params: int) -> np.ndarray:
    direction = rng.normal(size=n_params)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("Generated a zero random direction.")
    return direction / norm


def orthogonal_unit_vector(rng: np.random.Generator, reference: np.ndarray) -> np.ndarray:
    direction = rng.normal(size=reference.shape)
    direction = direction - np.dot(direction, reference) * reference
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return orthogonal_unit_vector(rng, reference)
    return direction / norm


def run_short_local_trajectories(energy, n_params: int, args, rng: np.random.Generator):
    history_x = []
    history_e = []
    history_run = []
    history_eval = []
    results = []

    for run in range(args.starts):
        x0 = rng.uniform(-np.pi, np.pi, size=n_params)
        counter = {"n": 0}

        def objective(x):
            counter["n"] += 1
            e = energy(x)
            history_x.append(np.asarray(x, dtype=float).copy())
            history_e.append(e)
            history_run.append(run)
            history_eval.append(counter["n"])
            return e

        print(f"[visualize] short local run {run + 1}/{args.starts}")
        res = minimize(
            objective,
            x0,
            method=args.optimizer,
            options={
                "maxiter": args.short_maxiter,
                "disp": False,
            },
        )
        print(f"[visualize]   final E={res.fun:.12f}")
        results.append(res)

    return (
        np.asarray(history_x),
        np.asarray(history_e),
        np.asarray(history_run),
        np.asarray(history_eval),
        results,
    )


def evaluate_slice(energy, center: np.ndarray, u: np.ndarray, v: np.ndarray, a_values: np.ndarray, b_values: np.ndarray):
    values = np.empty((len(b_values), len(a_values)), dtype=float)
    for i, b in enumerate(b_values):
        print(f"[visualize]   slice row {i + 1}/{len(b_values)}")
        for j, a in enumerate(a_values):
            values[i, j] = energy(center + a * u + b * v)
    return values


def plot_slice(path: Path, a_values: np.ndarray, b_values: np.ndarray, values: np.ndarray, title: str, hf_energy: float):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    mesh = ax.contourf(a_values, b_values, values, levels=48, cmap="viridis")
    ax.contour(a_values, b_values, values, levels=[hf_energy], colors="white", linewidths=1.2)
    ax.scatter([0.0], [0.0], c="red", s=36, label="center")
    ax.set_xlabel("direction a")
    ax.set_ylabel("direction b")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.colorbar(mesh, ax=ax, label="total energy")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca(path: Path, x_history: np.ndarray, e_history: np.ndarray, run_history: np.ndarray, hf_x: np.ndarray, best_x: np.ndarray | None):
    import matplotlib.pyplot as plt

    centered = x_history - x_history.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T

    hf_coord = (hf_x - x_history.mean(axis=0)) @ components.T
    best_coord = None
    if best_x is not None:
        best_coord = (best_x - x_history.mean(axis=0)) @ components.T

    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=e_history, s=14, cmap="viridis", alpha=0.8)

    for run in np.unique(run_history):
        mask = run_history == run
        ax.plot(coords[mask, 0], coords[mask, 1], linewidth=0.8, alpha=0.35)

    ax.scatter([hf_coord[0]], [hf_coord[1]], c="red", s=55, marker="x", label="HF")
    if best_coord is not None:
        ax.scatter([best_coord[0]], [best_coord[1]], c="white", edgecolors="black", s=55, label="best")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Optimizer trajectories projected by PCA")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="total energy")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sample_random_points(energy, n_params: int, n_samples: int, rng: np.random.Generator):
    x_samples = np.empty((n_samples, n_params), dtype=float)
    e_samples = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        if (i + 1) % 100 == 0 or i == 0 or i + 1 == n_samples:
            print(f"[visualize] random sample {i + 1}/{n_samples}")
        x = rng.uniform(-np.pi, np.pi, size=n_params)
        x_samples[i] = x
        e_samples[i] = energy(x)

    return x_samples, e_samples


def plot_random_pca(path: Path, x_samples: np.ndarray, e_samples: np.ndarray, hf_x: np.ndarray, best_x: np.ndarray | None):
    import matplotlib.pyplot as plt

    centered = x_samples - x_samples.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T

    hf_coord = (hf_x - x_samples.mean(axis=0)) @ components.T
    best_coord = None
    if best_x is not None:
        best_coord = (best_x - x_samples.mean(axis=0)) @ components.T

    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=e_samples, s=8, cmap="viridis", alpha=0.65)
    ax.scatter([hf_coord[0]], [hf_coord[1]], c="red", s=55, marker="x", label="HF")
    if best_coord is not None:
        ax.scatter([best_coord[0]], [best_coord[1]], c="white", edgecolors="black", s=55, label="best")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Random parameter samples projected by PCA")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="total energy")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize QCMPS parameter-space slices and local optimizer trajectories.")
    parser.add_argument("fcidump_path", type=Path)
    parser.add_argument("--bond_qubits", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--block_type", choices=sorted(BLOCK_TYPES), default="LPC")
    parser.add_argument("--optimizer", type=str, default="COBYLA")
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--short_maxiter", type=int, default=150)
    parser.add_argument("--random_samples", type=int, default=5000)
    parser.add_argument("--grid_size", type=int, default=31)
    parser.add_argument("--slice_span", type=float, default=1.5)
    parser.add_argument("--good_x", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    block_type = BLOCK_TYPES[args.block_type]

    hamiltonian, nuclear_repulsion, n_spin_orbitals, n_spatial_orbitals, num_alpha, num_beta, _ = build_from_fcidump(args.fcidump_path)
    hf_occ = hf_occupancy(num_alpha, num_beta, n_spatial_orbitals)
    blocks, params = prepare_blocks(n_spin_orbitals, args.bond_qubits, block_type, args.layers, hf_occ)

    n_params = len(params)
    energy = make_energy_function(blocks, params, hamiltonian, nuclear_repulsion)
    
    x_hf = np.zeros(n_params)
    e_hf = energy(x_hf)
    best_x = None
    best_e = None
    
    if args.random_samples > 0:
        print("[visualize] sampling random parameter points")
        x_samples, e_samples = sample_random_points(energy, n_params, args.random_samples, rng)
        np.savez(
            args.output / "random_samples.npz",
            x_samples=x_samples,
            e_samples=e_samples,
            x_hf=x_hf,
            e_hf=e_hf,
            best_x=best_x if best_x is not None else np.array([]),
            best_e=best_e if best_e is not None else np.nan,
        )
        plot_random_pca(args.output / "pca_random_samples.png", x_samples, e_samples, x_hf, best_x)

    print(f"[visualize] HF E={e_hf:.12f}")

    x_history, e_history, run_history, eval_history, results = run_short_local_trajectories(energy, n_params, args, rng)

    if results:
        best_result = min(results, key=lambda res: res.fun)
        best_x = best_result.x.copy()
        best_e = float(best_result.fun)

    if args.good_x is not None:
        best_x = np.loadtxt(args.good_x).reshape(-1)
        best_e = energy(best_x)
        print(f"[visualize] supplied good_x E={best_e:.12f}")

    np.savez(
        args.output / "visualize_data.npz",
        x_history=x_history,
        e_history=e_history,
        run_history=run_history,
        eval_history=eval_history,
        x_hf=x_hf,
        e_hf=e_hf,
        best_x=best_x if best_x is not None else np.array([]),
        best_e=best_e if best_e is not None else np.nan,
    )

    if len(x_history) >= 3:
        plot_pca(args.output / "pca_trajectories.png", x_history, e_history, run_history, x_hf, best_x)

    grid = np.linspace(-args.slice_span, args.slice_span, args.grid_size)

    print("[visualize] evaluating random HF slice")
    u = random_unit_vector(rng, n_params)
    v = orthogonal_unit_vector(rng, u)
    values = evaluate_slice(energy, x_hf, u, v, grid, grid)
    plot_slice(args.output / "hf_random_slice.png", grid, grid, values, "Random 2D slice around HF", e_hf)

    if best_x is not None and np.linalg.norm(best_x - x_hf) > 1e-12:
        print("[visualize] evaluating HF-to-best slice")
        u = best_x - x_hf
        distance = np.linalg.norm(u)
        u = u / distance
        v = orthogonal_unit_vector(rng, u)
        a_values = np.linspace(-0.25 * distance, 1.25 * distance, args.grid_size)
        b_values = np.linspace(-args.slice_span, args.slice_span, args.grid_size)
        values = evaluate_slice(energy, x_hf, u, v, a_values, b_values)
        plot_slice(args.output / "hf_to_best_slice.png", a_values, b_values, values, "2D slice from HF toward best point", e_hf)

    print(f"[visualize] wrote outputs to {args.output}")


if __name__ == "__main__":
    main()
