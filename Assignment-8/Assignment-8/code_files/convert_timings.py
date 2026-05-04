import pandas as pd

cases = ["a", "b", "c", "d", "e"]

for case in cases:
    # Read timing file
    df = pd.read_csv(f"timings_{case}.csv")
    
    # Calculate total cores (mpi * omp)
    df["Cores"] = df["mpi"] * df["omp"]
    
    # Rename total column to TotalTime
    df["TotalTime"] = df["total"]
    
    # Select relevant columns
    result = df[["Cores", "TotalTime", "mpi", "omp", "interp", "reduce", "broadcast", "norm", "mover", "denorm"]]
    
    # Save as results file
    result.to_csv(f"results_{case}.csv", index=False)
    
    print(f"Converted timings_{case}.csv -> results_{case}.csv")

print("\nConversion complete!")
