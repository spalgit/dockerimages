import os
import pandas as pd

# Set the directory containing the CSV files
directory = 'CDD_Article_Reduced_Graph_4'  # Change this to your path

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize a list to hold dataframes
df_list = []

# Read and append each CSV to the list
for file in csv_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    df_list.append(df)

# Concatenate all dataframes
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('combined_Reduced_Graph_CDD_Article_Reduced_Graph_4.csv', index=False)

print("All CSV files have been combined successfully!")
