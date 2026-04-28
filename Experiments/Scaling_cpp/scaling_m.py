import subprocess, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINARY = os.path.join(os.path.dirname(__file__),
                      "../../Implementations/AK-SVD-Parallel/build/main")
HERE = os.path.dirname(os.path.abspath(__file__))

M_SIZES = [5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 250000]

means = []
for M in M_SIZES:
    out = subprocess.run(
        [BINARY, "--synth", str(M)],
        capture_output=True, text=True,
        cwd=os.path.dirname(BINARY)
    ).stdout
    m = re.search(r"mean\s*:\s*([\d.]+)", out)
    mean = float(m.group(1)) if m else float("nan")
    means.append(mean)
    print(f"M={M:>7d}  mean={mean:.4f}s")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(M_SIZES, means, marker="o")
ax.set_xlabel("M (number of signals)")
ax.set_ylabel("Mean time (s)")
ax.set_title("AK-SVD Runtime vs Dataset Size")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()

out_path = os.path.join(HERE, "scaling_m_1.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
