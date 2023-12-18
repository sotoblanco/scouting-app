import os
import pandas as pd
import numpy as np
import pyarrow.feather as feather

# Read the CSV file
df = pd.read_csv('ENG1_GPT2_clean.csv')

# Calculate the size of the original DataFrame
original_size = df.memory_usage(deep=True).sum()

# Identify and compress the data types of the columns
for col in df.columns:
    if df[col].dtype == np.int64:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    elif df[col].dtype == np.float64:
        df[col] = pd.to_numeric(df[col], downcast='float')
    elif df[col].dtype == object:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')

# Calculate the size of the compressed DataFrame
compressed_size = df.memory_usage(deep=True).sum()

# Calculate the compression ratio
compression_ratio = original_size / compressed_size

# Store the DataFrame in a feather file
feather_file = 'compressed_data.feather'
feather.write_feather(df, feather_file)

print(f"Original Size: {original_size} bytes")
print(f"Compressed Size: {compressed_size} bytes")
print(f"Compression Ratio: {compression_ratio}")