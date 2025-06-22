# 0. Introduction
# Analysis of Climate Change data to predict CO2 emissions from country-specific parameters

# 1. Notebook Setup - Libraries and Data Import
import pandas as pd
import numpy as np

# Load the dataset
orig_data_file = r"climate_change_download_0.xls"
data_sheet = "Data"
data_orig = pd.read_excel(io=orig_data_file, sheet_name=data_sheet)

# 2. Global Data Overview
print("Original dataset shape:")
print(data_orig.shape)

print("\nColumn names:")
print(data_orig.columns)

print("\nUnique values in 'SCALE' column:")
print(data_orig['SCALE'].unique())

# 3. Definition of Initial Project Goals
# Goal: Clean and prepare the dataset for predictive analysis on CO2 emissions

# 4. Data Cleaning

# Remove rows where 'SCALE' or 'Decimals' contain 'Text'
data_clean = data_orig.copy()
data_clean = data_clean[data_clean['SCALE'] != 'Text']
data_clean = data_clean[data_clean['Decimals'] != 'Text']

# Replace '.' and empty strings with NaN
data_clean.iloc[:, 2:] = data_clean.iloc[:, 2:].replace({'': np.nan, '..': np.nan})

# Convert all values to numeric where applicable
data_clean2 = data_clean.applymap(lambda x: pd.to_numeric(x, errors='ignore'))

print("\nData types after conversion:")
print(data_clean2.dtypes)

# Optional: Rename columns for ease (example shown)
# data_clean2.rename(columns={"Country name": "Country", "Series name": "Indicator"}, inplace=True)

# Remove empty rows/columns
data_clean2.dropna(how='all', axis=0, inplace=True)
data_clean2.dropna(how='all', axis=1, inplace=True)

# 5. Data Frame Transformation - Melting
# Reshape data: from wide to long format
id_vars = ['Country code', 'Country name', 'Series code', 'Series name', 'SCALE', 'Decimals']
value_vars = [col for col in data_clean2.columns if col not in id_vars]

data_melted = pd.melt(data_clean2, id_vars=id_vars, value_vars=value_vars,
                      var_name='Year', value_name='Value')

# 6. Removal of Missing Values
# Detect & drop rows with missing 'Value' (target column)
data_melted.dropna(subset=['Value'], inplace=True)

# 7. Export the cleaned data to file
data_melted.to_csv("clean_climate_data.csv", index=False)

print("\nCleaned data saved to 'clean_climate_data.csv'")
print("Final cleaned shape:", data_melted.shape)
