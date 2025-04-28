import pandas as pd

# Exploratory Data Analysis has shown pcv, sc and hemo to be the most relevant columns for predicting ckd
# Creating a new dataframe with only the relevant columns
chronic_kidney_data = pd.read_csv('chronic_kidney_data_with_id.csv')

relevant_columns = ['id', 'pcv', 'sc', 'hemo', 'class']
chronic_kidney_data_relevant = chronic_kidney_data[relevant_columns]

# removing rows with missing values in pcv, sc and hemo
chronic_kidney_data_relevant = chronic_kidney_data_relevant.dropna(subset=['pcv', 'sc', 'hemo'])

# Save the new dataframe to a CSV file
chronic_kidney_data_relevant.to_csv('chronic_kidney_data_relevant.csv', index=False)

print("Dataset with relevant columns saved to chronic_kidney_data_relevant.csv")