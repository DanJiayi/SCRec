#!/usr/bin/env python3
"""
Extracts metric sweeps from grid_test_results_Toys_and_Games.txt.

For each of the three hyper-parameters (contrastive_alpha, manifold_beta,
manifold_c) we keep the other two fixed at 0.5, 0.2, 0.2 respectively and emit
rows with schema:
    parameter, value, recall@5, ndcg@5, recall@10, ndcg@10
"""

import argparse
import ast
import re
from pathlib import Path


LINE_REGEX = re.compile(
    r"contrastive_alpha=([0-9.]+)\s+manifold_beta=([0-9.]+)\s+manifold_c=([0-9.]+)\s*\|\s*INFO:root:Test Results: OrderedDict\((\{.*\})\)"
)


def parse_results(path: Path):
    records = []
    for line in path.read_text().splitlines():
        m = LINE_REGEX.search(line)
        if not m:
            continue
        ca, mb, mc, metrics_blob = m.groups()
        metrics_dict = ast.literal_eval(metrics_blob)
        records.append(
            {
                "contrastive_alpha": float(ca),
                "manifold_beta": float(mb),
                "manifold_c": float(mc),
                "recall@5": float(metrics_dict["recall@5"]),
                "ndcg@5": float(metrics_dict["ndcg@5"]),
                "recall@10": float(metrics_dict["recall@10"]),
                "ndcg@10": float(metrics_dict["ndcg@10"]),
            }
        )
    return records


def collect(rows, param, fixed_cond):
    for record in rows:
        if all(abs(record[key] - val) < 1e-9 for key, val in fixed_cond.items()):
            yield {
                "value": record[param],
                "recall@5": record["recall@5"],
                "ndcg@5": record["ndcg@5"],
                "recall@10": record["recall@10"],
                "ndcg@10": record["ndcg@10"],
            }


def main():
    parser = argparse.ArgumentParser(description="Summarize grid metrics.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/root/test/gr/grid_test_results_Beauty.txt"),
        help="Path to the grid results text file.",
    )
    parser.add_argument("--category", default="Beauty", help="Category label for reference.")
    parser.add_argument("--fixed_manifold_beta", type=float, default=0.2, help="Fixed manifold_beta when sweeping contrastive_alpha.")
    parser.add_argument("--fixed_manifold_c", type=float, default=0.5, help="Fixed manifold_c when sweeping contrastive_alpha or manifold_beta.")
    parser.add_argument("--fixed_contrastive_alpha", type=float, default=0.5, help="Fixed contrastive_alpha when sweeping manifold_beta or manifold_c.")
    args = parser.parse_args()

    records = parse_results(args.results)
    if not records:
        raise SystemExit(f"No valid lines parsed from {args.results}")

    sweeps = [
        ("contrastive_alpha", {"manifold_beta": args.fixed_manifold_beta, "manifold_c": args.fixed_manifold_c}),
        ("manifold_beta", {"contrastive_alpha": args.fixed_contrastive_alpha, "manifold_c": args.fixed_manifold_c}),
        ("manifold_c", {"contrastive_alpha": args.fixed_contrastive_alpha, "manifold_beta": args.fixed_manifold_beta}),
    ]

    print(f"# category={args.category}")
    has_output = False
    for param, cond in sweeps:
        rows = list(collect(records, param, cond))
        if not rows:
            continue
        has_output = True
        print(f"\n## Parameter sweep: {param} (fixed {', '.join(f'{k}={v}' for k, v in cond.items())})")
        print("value,recall@5,ndcg@5,recall@10,ndcg@10")
        for row in sorted(rows, key=lambda r: r["value"]):
            print(
                f"{row['value']:.6g},"
                f"{row['recall@5']:.8f},{row['ndcg@5']:.8f},"
                f"{row['recall@10']:.8f},{row['ndcg@10']:.8f}"
            )

    if not has_output:
        raise SystemExit("No rows matched the specified fixed-value conditions.")


if __name__ == "__main__":
    main()
