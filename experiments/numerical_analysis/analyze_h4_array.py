from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the completed H4 QCMPS array experiment."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=HERE / "h4_array_results",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "systems" / "H4" / "H4_reference.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "h4_array_analysis",
    )
    return parser.parse_args()


def load_rows(results_dir: Path, hf_energy: float, reference_energy: float) -> list[dict]:
    correlation_energy = hf_energy - reference_energy
    rows = []

    for path in sorted(results_dir.glob("h4_q*_seed*_layer*.json")):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)

        sweep = record.get("sweep")
        global_stage = record.get("global", record.get("global_polish"))

        energy = float(record["total_energy"])
        below_hf = hf_energy - energy
        error = energy - reference_energy

        rows.append(
            {
                "source": path.name,
                "system": record["system"],
                "seed": int(record["seed"]),
                "n_bond_qubits": int(record["n_bond_qubits"]),
                "bond_dimension": int(record["bond_dimension"]),
                "n_layers": int(record["n_layers"]),
                "n_parameters": int(record["n_parameters"]),
                "total_energy_hartree": energy,
                "below_hf_mha": 1000.0 * below_hf,
                "error_to_reference_mha": 1000.0 * error,
                "correlation_recovered_percent": 100.0 * below_hf / correlation_energy,
                "sweep_converged": None if sweep is None else bool(sweep["converged"]),
                "sweep_nfev": 0 if sweep is None else int(sweep["n_function_evaluations"]),
                "sweep_nit": 0 if sweep is None else int(sweep["n_iterations"]),
                "sweep_message": "" if sweep is None else str(sweep["message"]),
                "global_converged": bool(global_stage["converged"]),
                "global_nfev": int(global_stage["n_function_evaluations"]),
                "global_nit": int(global_stage["n_iterations"]),
                "global_message": str(global_stage["message"]),
                "total_nfev": (
                    (0 if sweep is None else int(sweep["n_function_evaluations"]))
                    + int(global_stage["n_function_evaluations"])
                ),
            }
        )

    if not rows:
        raise ValueError(f"No H4 array JSON files found in {results_dir}.")

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    aggregated = []
    keys = sorted({(row["n_bond_qubits"], row["n_layers"]) for row in rows})

    for n_bond_qubits, n_layers in keys:
        group = [
            row
            for row in rows
            if row["n_bond_qubits"] == n_bond_qubits
            and row["n_layers"] == n_layers
        ]
        energies = [row["total_energy_hartree"] for row in group]
        errors = [row["error_to_reference_mha"] for row in group]
        correlations = [row["correlation_recovered_percent"] for row in group]
        best = min(group, key=lambda row: row["total_energy_hartree"])

        aggregated.append(
            {
                "n_bond_qubits": n_bond_qubits,
                "bond_dimension": 2**n_bond_qubits,
                "n_layers": n_layers,
                "n_seeds": len(group),
                "mean_energy_hartree": fmean(energies),
                "energy_std_mha": 1000.0 * pstdev(energies),
                "best_energy_hartree": min(energies),
                "worst_energy_hartree": max(energies),
                "mean_error_to_reference_mha": fmean(errors),
                "best_error_to_reference_mha": min(errors),
                "mean_correlation_recovered_percent": fmean(correlations),
                "best_correlation_recovered_percent": max(correlations),
                "best_seed": best["seed"],
            }
        )

    return aggregated


def summarize_trajectories(rows: list[dict]) -> list[dict]:
    summaries = []
    keys = sorted({(row["n_bond_qubits"], row["seed"]) for row in rows})

    for n_bond_qubits, seed in keys:
        trajectory = sorted(
            (
                row
                for row in rows
                if row["n_bond_qubits"] == n_bond_qubits
                and row["seed"] == seed
            ),
            key=lambda row: row["n_layers"],
        )
        first = trajectory[0]
        final = trajectory[-1]

        summaries.append(
            {
                "n_bond_qubits": n_bond_qubits,
                "bond_dimension": first["bond_dimension"],
                "seed": seed,
                "initial_energy_hartree": first["total_energy_hartree"],
                "final_energy_hartree": final["total_energy_hartree"],
                "layer_1_to_4_gain_mha": 1000.0
                * (first["total_energy_hartree"] - final["total_energy_hartree"]),
                "final_error_to_reference_mha": final["error_to_reference_mha"],
                "final_correlation_recovered_percent": final[
                    "correlation_recovered_percent"
                ],
                "cumulative_nfev": sum(row["total_nfev"] for row in trajectory),
            }
        )

    return summaries


def make_overview_plot(
    path: Path,
    rows: list[dict],
    hf_energy: float,
    reference_energy: float,
) -> None:
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    markers = {0: "o", 1: "s"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    for q in sorted({row["n_bond_qubits"] for row in rows}):
        for seed in sorted({row["seed"] for row in rows}):
            group = sorted(
                (
                    row
                    for row in rows
                    if row["n_bond_qubits"] == q and row["seed"] == seed
                ),
                key=lambda row: row["n_layers"],
            )
            ax.plot(
                [row["n_layers"] for row in group],
                [row["total_energy_hartree"] for row in group],
                marker=markers[seed],
                color=colors[q],
                linestyle="-" if seed == 0 else "--",
                label=f"q={q}, D={2**q}, seed={seed}",
            )
    ax.axhline(hf_energy, color="black", linestyle=":", label="HF")
    ax.axhline(reference_energy, color="tab:red", linestyle=":", label="CASCI")
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Total energy [Ha]")
    ax.set_title("Energy trajectories")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    for q in sorted({row["n_bond_qubits"] for row in rows}):
        for seed in sorted({row["seed"] for row in rows}):
            group = sorted(
                (
                    row
                    for row in rows
                    if row["n_bond_qubits"] == q and row["seed"] == seed
                ),
                key=lambda row: row["n_layers"],
            )
            ax.plot(
                [row["n_layers"] for row in group],
                [row["correlation_recovered_percent"] for row in group],
                marker=markers[seed],
                color=colors[q],
                linestyle="-" if seed == 0 else "--",
            )
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Recovered correlation energy [%]")
    ax.set_title("Correlation recovery")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for q in sorted({row["n_bond_qubits"] for row in rows}):
        group = sorted(
            (row for row in rows if row["n_bond_qubits"] == q),
            key=lambda row: (row["seed"], row["n_parameters"]),
        )
        for seed in sorted({row["seed"] for row in group}):
            seed_group = [row for row in group if row["seed"] == seed]
            ax.plot(
                [row["n_parameters"] for row in seed_group],
                [row["error_to_reference_mha"] for row in seed_group],
                marker=markers[seed],
                color=colors[q],
                linestyle="-" if seed == 0 else "--",
                label=f"q={q}, seed={seed}",
            )
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Error to CASCI [mHa]")
    ax.set_title("Accuracy versus ansatz size")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    for q in sorted({row["n_bond_qubits"] for row in rows}):
        for seed in sorted({row["seed"] for row in rows}):
            group = sorted(
                (
                    row
                    for row in rows
                    if row["n_bond_qubits"] == q and row["seed"] == seed
                ),
                key=lambda row: row["n_layers"],
            )
            ax.plot(
                [row["n_layers"] for row in group],
                [row["total_nfev"] for row in group],
                marker=markers[seed],
                color=colors[q],
                linestyle="-" if seed == 0 else "--",
            )
    ax.set_yscale("log")
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Function evaluations")
    ax.set_title("Optimization cost per stage")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.25, which="both")

    figure.suptitle("H4 QCMPS array experiment", fontsize=15)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def make_report(
    path: Path,
    rows: list[dict],
    aggregated: list[dict],
    trajectories: list[dict],
    hf_energy: float,
    reference_energy: float,
) -> None:
    best = min(rows, key=lambda row: row["total_energy_hartree"])
    layer4 = [row for row in aggregated if row["n_layers"] == 4]
    global_failures = [row for row in rows if not row["global_converged"]]
    sweep_rows = [row for row in rows if row["sweep_converged"] is not None]
    sweep_failures = [row for row in sweep_rows if not row["sweep_converged"]]
    stop_reasons = Counter(row["global_message"] for row in global_failures)

    lines = [
        "# Analýza H4 QCMPS array experimentu",
        "",
        "## Reference",
        "",
        f"- Celková HF energie: `{hf_energy:.12f} Ha`",
        f"- Celková CASCI energie: `{reference_energy:.12f} Ha`",
        f"- Korelační energie: `{1000.0 * (hf_energy - reference_energy):.6f} mHa`",
        f"- Počet analyzovaných běhů: `{len(rows)}`",
        "",
        "## Nejlepší výsledek",
        "",
        (
            f"Nejlepší výsledek array experimentu má `q={best['n_bond_qubits']}` "
            f"(`D={best['bond_dimension']}`), seed `{best['seed']}`, "
            f"`{best['n_layers']}` vrstvy a `{best['n_parameters']}` parametrů."
        ),
        "",
        f"- Celková energie: `{best['total_energy_hartree']:.12f} Ha`",
        f"- Chyba vůči CASCI: `{best['error_to_reference_mha']:.6f} mHa`",
        f"- Získaná korelační energie: `{best['correlation_recovered_percent']:.3f} %`",
        "",
        "## Porovnání čtyřvrstvých ansatzů",
        "",
        "| Bond qubity | Bond dimension | Průměrná energie [Ha] | Směrodatná odchylka seedů [mHa] | Nejlepší energie [Ha] | Nejlepší seed | Nejlepší korelace [%] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in layer4:
        lines.append(
            f"| {row['n_bond_qubits']} | {row['bond_dimension']} | "
            f"{row['mean_energy_hartree']:.12f} | {row['energy_std_mha']:.6f} | "
            f"{row['best_energy_hartree']:.12f} | {row['best_seed']} | "
            f"{row['best_correlation_recovered_percent']:.3f} |"
        )

    lines.extend(
        [
            "",
            "Byly použity pouze dva seedy, takže průměr a směrodatná odchylka jsou jen popisné hodnoty, nikoliv spolehlivý statistický odhad.",
            "",
            "## Celé trajektorie od jedné do čtyř vrstev",
            "",
            "| Bond qubity | Bond dimension | Seed | Zisk L1 → L4 [mHa] | Finální chyba [mHa] | Finální korelace [%] | Kumulativní evaluace |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in trajectories:
        lines.append(
            f"| {row['n_bond_qubits']} | {row['bond_dimension']} | {row['seed']} | "
            f"{row['layer_1_to_4_gain_mha']:.6f} | "
            f"{row['final_error_to_reference_mha']:.6f} | "
            f"{row['final_correlation_recovered_percent']:.3f} | "
            f"{row['cumulative_nfev']} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostika konvergence",
            "",
            f"- Globální etapy ukončené neúspěchem nebo limitem: `{len(global_failures)}/{len(rows)}`",
            f"- Sweep etapy ukončené maximálním počtem sweepů: `{len(sweep_failures)}/{len(sweep_rows)}`",
        ]
    )

    for reason, count in stop_reasons.most_common():
        lines.append(f"- `{count}` globálních běhů: {reason}")

    lines.extend(
        [
            "",
            "Produktivní trajektorie často skončily na limitu evaluací nebo sweepů. Několik formálně zkonvergovaných běhů naopak zůstalo blízko HF. Příznak úspěchu optimalizátoru proto sám o sobě nevypovídá o fyzikální kvalitě řešení.",
            "",
            "## Hlavní závěry",
            "",
            "1. Výsledek silně závisí na seedu. Stejně velký ansatz může zůstat u HF, nebo získat významnou část korelační energie.",
            "2. Vyšší bond dimension nevede monotónně k lepší optimalizované energii. Větší parametrické prostory se současným numerickým gradientem optimalizují hůře.",
            "3. Přidávání vrstev pomáhá pouze podél produktivních optimalizačních trajektorií. Jiné seedy stagnují téměř okamžitě navzdory vyšší expresivitě.",
            "4. Nejlepší array běh má stále velkou chybu vůči CASCI. Z těchto dat nelze tvrdit, že je zkonvergovaná expresivita ani optimalizace.",
            "5. Procenta korelační energie jsou předběžná, dokud nebude u SO(4) ansatzu ověřena střední hodnota a variance částicového čísla.",
            "",
            "## Vygenerované soubory",
            "",
            "- `runs.csv`: jeden řádek pro každý seed, bond dimension a počet vrstev.",
            "- `aggregated.csv`: průměr, rozptyl a nejlepší výsledek pro každou bond dimension a počet vrstev.",
            "- `trajectories.csv`: souhrn celé adaptivní trajektorie každého seedu.",
            "- `overview.png`: grafy energie, korelační energie, přesnosti a ceny optimalizace.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.reference.open(encoding="utf-8") as file:
        reference = json.load(file)

    hf_energy = float(reference["scf_total_energy_hartree"])
    reference_energy = float(reference["reference_total_energy_hartree"])

    rows = load_rows(args.results_dir, hf_energy, reference_energy)
    aggregated = aggregate_rows(rows)
    trajectories = summarize_trajectories(rows)

    write_csv(args.output_dir / "runs.csv", rows)
    write_csv(args.output_dir / "aggregated.csv", aggregated)
    write_csv(args.output_dir / "trajectories.csv", trajectories)
    make_overview_plot(
        args.output_dir / "overview.png",
        rows,
        hf_energy,
        reference_energy,
    )
    make_report(
        args.output_dir / "report.md",
        rows,
        aggregated,
        trajectories,
        hf_energy,
        reference_energy,
    )

    print(f"Analyzed {len(rows)} runs.")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
