import subprocess, re, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINARY = os.path.join(os.path.dirname(__file__),
                              "../../Implementations/AK-SVD-Parallel/build/main")
HERE = os.path.dirname(os.path.abspath(__file__))

M = 400000
THREADS = [1, 2, 4, 8, 16, 32, 64, 128]

means = []
losses = []
throughputs = []

for p in THREADS:
        env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(p)

                out = subprocess.run(
                                [BINARY, "--synth", str(M)],
                                        capture_output=True, text=True,
                                                env=env, cwd=os.path.dirname(BINARY)
                                                    ).stdout

                    # Parse runtime
                        m_time = re.search(r"mean\s*:\s*([\d.]+)", out)
                            mean = float(m_time.group(1)) if m_time else float("nan")

                                # Parse loss (adjust regex if needed)
                                    m_loss = re.search(r"loss\s*:\s*([\d.eE+-]+)", out)
                                        loss = float(m_loss.group(1)) if m_loss else 0.0  # fallback

                                            throughput = M / mean if mean > 0 else float("nan")

                                                means.append(mean)
                                                    losses.append(loss)
                                                        throughputs.append(throughput)

                                                            print(f"P={p:3d}  mean={mean:.4f}s  throughput={throughput:.4f}")

                                                            # ---- Save CSV ----
                                                            csv_path = os.path.join(HERE, "ksvd_scaling_strong.csv")

                                                            with open(csv_path, "w", newline="") as f:
                                                                    writer = csv.writer(f)
                                                                        writer.writerow(["threads", "runtime", "loss", "throughput"])
                                                                            for i in range(len(THREADS)):
                                                                                        writer.writerow([THREADS[i], means[i], losses[i], throughputs[i]])

                                                                                        print(f"\nSaved CSV to {csv_path}")

                                                                                        # ---- Plot ----
                                                                                        ideal = [means[0] / p for p in THREADS]

                                                                                        fig, ax = plt.subplots(figsize=(7, 4))
                                                                                        ax.plot(THREADS, means, marker="o", label="Measured")
                                                                                        ax.plot(THREADS, ideal, linestyle="--", label="Ideal (1/P scaling)")

                                                                                        ax.set_xlabel("Threads (P)")
                                                                                        ax.set_ylabel("Runtime (s)")
                                                                                        ax.set_title("AK-SVD Strong Scaling (M = 400k)")
                                                                                        ax.set_xticks(THREADS)
                                                                                        ax.legend()
                                                                                        ax.grid(True, linestyle="--", alpha=0.5)

                                                                                        fig.tight_layout()

                                                                                        out_path = os.path.join(HERE, "strong_scaling.png")
                                                                                        fig.savefig(out_path, dpi=150)

                                                                                        print(f"Saved plot to {out_path}")
