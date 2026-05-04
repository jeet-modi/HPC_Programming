import pandas as pd
import matplotlib.pyplot as plt
import os

# Create graphs folder
os.makedirs("graphs", exist_ok=True)

cases = ["a", "b", "c", "d", "e"]
labels = {
    "a": "250x100, 0.9M",
    "b": "250x100, 5M",
    "c": "500x200, 3.6M",
    "d": "500x200, 20M",
    "e": "1000x400, 14M"
}


for case in cases:

    filename = f"results_{case}.csv"

    print(f"Processing {filename}")

    df = pd.read_csv(filename)

    df["Cores"] = pd.to_numeric(df["Cores"])
    df["TotalTime"] = pd.to_numeric(df["TotalTime"])

    # Get best per core
    best = df.loc[df.groupby("Cores")["TotalTime"].idxmin()]
    best = best.sort_values("Cores")

    # Save best table
    best.to_csv(f"graphs/best_{case}.csv", index=False)

    # ===============================
    # TIME vs CORES
    # ===============================
    plt.figure(figsize=(8,5))

    plt.plot(
        best["Cores"],
        best["TotalTime"],
        marker='o',
        linewidth=2
    )

    plt.xscale("log", base=2)
    plt.xticks([2,4,8,16,32,64],[2,4,8,16,32,64])

    plt.xlabel("Total Cores")
    plt.ylabel("Execution Time (sec)")
    plt.title(f"Case {case.upper()} - Time vs Cores")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"graphs/time_{case}.png", dpi=300)
    plt.close()

    # ===============================
    # SPEEDUP
    # ===============================
    plt.figure(figsize=(8,5))

    t1 = best[best["Cores"] == 2]["TotalTime"].values[0]
    best["Speedup"] = t1 / best["TotalTime"]

    plt.plot(
        best["Cores"],
        best["Speedup"],
        marker='o',
        linewidth=2
    )

    plt.xscale("log", base=2)
    plt.xticks([2,4,8,16,32,64],[2,4,8,16,32,64])

    plt.xlabel("Total Cores")
    plt.ylabel("Speedup")
    plt.title(f"Case {case.upper()} - Speedup vs Cores")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"graphs/speedup_{case}.png", dpi=300)
    plt.close()

    print(f"Graphs saved for case {case}")

print("\nAll graphs generated successfully.")

plt.figure(figsize=(10,6))

for case in cases:

    df = pd.read_csv(f"results_{case}.csv")

    df["Cores"] = pd.to_numeric(df["Cores"])
    df["TotalTime"] = pd.to_numeric(df["TotalTime"])

    best = df.loc[df.groupby("Cores")["TotalTime"].idxmin()]
    best = best.sort_values("Cores")

    plt.plot(
        best["Cores"],
        best["TotalTime"],
        marker='o',
        linewidth=2,
        label=labels[case]
    )

plt.xscale("log", base=2)
plt.xticks([2,4,8,16,32,64],[2,4,8,16,32,64])

plt.xlabel("Total Cores")
plt.ylabel("Execution Time (sec)")
plt.title("Execution Time vs Cores (All Cases)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("graphs/time_all_cases.png", dpi=300)
plt.close()

# =========================================
# SPEEDUP (ALL CASES)
# =========================================
plt.figure(figsize=(10,6))

for case in cases:

    df = pd.read_csv(f"results_{case}.csv")

    df["Cores"] = pd.to_numeric(df["Cores"])
    df["TotalTime"] = pd.to_numeric(df["TotalTime"])

    best = df.loc[df.groupby("Cores")["TotalTime"].idxmin()]
    best = best.sort_values("Cores")

    t1 = best[best["Cores"] == 2]["TotalTime"].values[0]
    best["Speedup"] = t1 / best["TotalTime"]

    plt.plot(
        best["Cores"],
        best["Speedup"],
        marker='o',
        linewidth=2,
        label=labels[case]
    )

plt.xscale("log", base=2)
plt.xticks([2,4,8,16,32,64],[2,4,8,16,32,64])

plt.xlabel("Total Cores")
plt.ylabel("Speedup")
plt.title("Speedup vs Cores (All Cases)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("graphs/speedup_all_cases.png", dpi=300)
plt.close()

print("Combined graphs generated successfully.")
