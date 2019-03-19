import pandas as pd

# Sample data to play with and clean.
data = {
    'age': [27, 50, 34, None, None, None],
    'gender': ['f', 'f', 'f', 'm', 'm', None],
    'height' : [64, None, 71, 66, 68, None],
    'weight' : [140, None, 130, 110, 160, None],
}
df = pd.DataFrame(data)

# Full dataset.
print(df)

# Drop all rows that have any missing values in any column.
print(df.dropna()) 

# Drop only rows where all values are missing.
print(df.dropna(how='all'))

# Drop only rows where more than two values are missing.
print(df.dropna(thresh=2))

# Drop all rows that have any missing values in the 'gender' or 'height' columns.
print(df.dropna(subset=['gender','height']))

# Your turn. Write code below to drop rows where both height and weight
# are missing and print the result.
