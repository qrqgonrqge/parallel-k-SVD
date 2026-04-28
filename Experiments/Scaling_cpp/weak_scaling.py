import subprocess, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINARY = os.path.join(os.path.dirname(__file__),
                      "../../Implementations/AK-SVD-Parallel/build/main")
HERE = os.path.dirname(os.path.abspath(__file__))

M_BASE = 10000  # signals per thread
THREADS = [1, 2, 4, 6, 8, 12]

means = []
for p in THREADS:
    M = M_BASE * p
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(p)
    out = subprocess.run(
        [BINARY, "--synth", str(M)],
        capture_output=True, text=True,
        env=env, cwd=os.path.dirname(BINARY)
    ).stdout
    m = re.search(r"mean\s*:\s*([\d.]+)", out)
    mean = float(m.group(1)) if m else float("nan")
    means.append(mean)
    print(f"P={p:2d}  M={M:>7d}  mean={mean:.4f}s")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(THREADS, means, marker="o", label="Measured")
ax.axhline(means[0], linestyle="--", color="grey", label="Ideal (flat)")
ax.set_xlabel("Threads (P)  —  M = 10k × P")
ax.set_ylabel("Mean time (s)")
ax.set_title("AK-SVD Weak Scaling  (fixed M/P = 10k)")
ax.set_xticks(THREADS)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()

out_path = os.path.join(HERE, "weak_scaling.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
