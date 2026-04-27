import subprocess, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINARY = os.path.join(os.path.dirname(__file__),
                      "../../Implementations/AK-SVD-Parallel/build/main")
HERE = os.path.dirname(os.path.abspath(__file__))

results = {}
for p in range(1, 13):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(p)
    out = subprocess.run([BINARY], capture_output=True, text=True, env=env,
                         cwd=os.path.dirname(BINARY)).stdout
    m = re.search(r"mean\s*:\s*([\d.]+)", out)
    mean = float(m.group(1)) if m else float("nan")
    results[p] = mean
    print(f"P={p:2d}  mean={mean:.4f}s  speedup={results[1]/mean:.2f}x")

ps    = list(results.keys())
means = list(results.values())
speedups = [results[1] / t for t in means]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(ps, means, marker="o")
ax1.set_xlabel("Threads (P)")
ax1.set_ylabel("Mean time (s)")
ax1.set_title("Runtime vs Threads")
ax1.set_xticks(ps)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(ps, speedups, marker="o", label="Measured")
ax2.plot(ps, ps, linestyle="--", color="grey", label="Ideal")
ax2.set_xlabel("Threads (P)")
ax2.set_ylabel("Speedup")
ax2.set_title("Strong Scaling Speedup")
ax2.set_xticks(ps)
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
out_path = os.path.join(HERE, "strong_scaling.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
