import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('dataset 4.csv')

# Make sure there are at least 3 rows
assert df.shape[0] >= 3, "Dataset has less than 3 rows"

# Introduce missing values in the "Course" column
indices = [0, 1, 2]  # Change this to the indices of your choice
df.loc[indices, 'Course'] = np.nan

# Save the modified DataFrame to a new CSV file
df.to_csv('dataset4_with_missing_values.csv', index=False)