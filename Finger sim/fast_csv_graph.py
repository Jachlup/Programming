from pathlib import Path
import csv

import matplotlib.pyplot as plt


def load_node_positions(csv_path: Path):
	points_by_rod = {}
	with csv_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rod = row["rod"]
			x = float(row["x"])
			y = float(row["y"])
			points_by_rod.setdefault(rod, {"x": [], "y": []})
			points_by_rod[rod]["x"].append(x)
			points_by_rod[rod]["y"].append(y)
	return points_by_rod


def plot_xy_scatter(points_by_rod, output_path: Path):
	fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

	for rod, coords in points_by_rod.items():
		ax.scatter(coords["x"], coords["y"], s=40, alpha=0.9, label=rod)

	ax.set_title("Node Positions After Simulation (XY Scatter)")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, linestyle="--", alpha=0.4)
	ax.legend()

	fig.tight_layout()
	fig.savefig(output_path)
	plt.show()


def main():
	script_dir = Path(__file__).resolve().parent
	csv_path = script_dir / "node_positions_after_simulation.csv"
	output_path = script_dir / "node_positions_xy_scatter.png"

	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	points_by_rod = load_node_positions(csv_path)
	plot_xy_scatter(points_by_rod, output_path)
	print(f"Saved XY scatter plot to: {output_path}")


if __name__ == "__main__":
	main()
