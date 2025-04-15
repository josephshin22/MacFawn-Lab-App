import pandas as pd

# Load the TSV file
input_file = "./static/data/measurements.tsv"  # Replace with your TSV file name
# Replace with desired CSV file name
output_file = "./static/data/measurements.csv"

# Read the TSV file
df = pd.read_csv(input_file, sep="\t")

# Save as a CSV file
df.to_csv(output_file, index=False)

print(f"File converted successfully and saved as {output_file}")
