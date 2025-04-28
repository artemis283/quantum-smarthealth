from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset 
chronic_kidney_disease = fetch_ucirepo(id=336)

# Data (as pandas dataframes) 
X = chronic_kidney_disease.data.features  # Features (X)
y = chronic_kidney_disease.data.targets    # Target (y)

# Combine features (X) and targets (y)
chronic_kidney_data = pd.concat([X, y], axis=1)

# Add a unique identifier column (using a simple count from 1 to n)
chronic_kidney_data['id'] = range(1, len(chronic_kidney_data) + 1)

# Reorder columns if you want the 'id' to be the first column
chronic_kidney_data = chronic_kidney_data[['id'] + [col for col in chronic_kidney_data.columns if col != 'id']]

# Save the data to a CSV file
chronic_kidney_data.to_csv('chronic_kidney_data_with_id.csv', index=False)

print("Dataset with unique ID saved to chronic_kidney_data_with_id.csv")




