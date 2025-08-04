import pandas as pd

filename = 'twins_model_SOL0182.csv'
pressure = pd.read_csv(filename)

# Get the unique pressure frequencies
a = pd.unique(pressure['WIND_FREQUENCY'])

print(a)

b1 = pressure[pressure['WIND_FREQUENCY'] == a[1]]
b2 = pressure[pressure['WIND_FREQUENCY'] == a[2]]

# Initialize a list to store the downsampled rows
downsampled_rows = []

# Add the first row manually with the updated frequency
first_row = b2.iloc[0].copy()
first_row['WIND_FREQUENCY'] = 0.1
downsampled_rows.append(first_row)

# Calculate the moving average for every 5th row starting from index 4
for i in range(5, len(b2), 5):
    # Create the downsampled row as a Series
    downsampled_row = pd.Series({
        'AOBT': b2.iloc[i]['AOBT'],
        'SCLK': b2.iloc[i]['SCLK'],
        'LMST': b2.iloc[i]['LMST'],
        'LTST': b2.iloc[i]['LTST'],
        'UTC': b2.iloc[i]['UTC'],
        'HORIZONTAL_WIND_SPEED': b2.iloc[i-4:i+1]['HORIZONTAL_WIND_SPEED'].mean(),
        'VERTICAL_WIND_SPEED': b2.iloc[i-4:i+1]['VERTICAL_WIND_SPEED'].mean(),
        'WIND_DIRECTION': b2.iloc[i-4:i+1]['WIND_DIRECTION'].mean(),
        'WIND_FREQUENCY': 0.1,
    })
    downsampled_rows.append(downsampled_row)

# Convert the list of downsampled Series objects to a DataFrame
downsampled_df = pd.DataFrame(downsampled_rows)

# Combine b1 and the downsampled DataFrame
new_twins = pd.concat([b1, downsampled_df], ignore_index=True)

new_twins.to_csv(filename, index=False)
#______________________________________________________________________________________________________

filename = 'twins_model_SOL0230.csv'
pressure = pd.read_csv(filename)

# Get the unique pressure frequencies
a = pd.unique(pressure['WIND_FREQUENCY'])

print(a)

b1 = pressure[pressure['WIND_FREQUENCY'] == a[0]]
b2 = pressure[pressure['WIND_FREQUENCY'] == a[1]]

# Initialize a list to store the downsampled rows
downsampled_rows = []

# Add the first row manually with the updated frequency
first_row = b2.iloc[0].copy()
first_row['WIND_FREQUENCY'] = 0.5
downsampled_rows.append(first_row)

# Calculate the moving average for every 5th row starting from index 4
for i in range(2, len(b2), 2):
    # Create the downsampled row as a Series
    downsampled_row = pd.Series({
        'AOBT': b2.iloc[i]['AOBT'],
        'SCLK': b2.iloc[i]['SCLK'],
        'LMST': b2.iloc[i]['LMST'],
        'LTST': b2.iloc[i]['LTST'],
        'UTC': b2.iloc[i]['UTC'],
        'HORIZONTAL_WIND_SPEED': b2.iloc[i-2:i+1]['HORIZONTAL_WIND_SPEED'].mean(),
        'VERTICAL_WIND_SPEED': b2.iloc[i-2:i+1]['VERTICAL_WIND_SPEED'].mean(),
        'WIND_DIRECTION': b2.iloc[i-2:i+1]['WIND_DIRECTION'].mean(),
        'WIND_FREQUENCY': 0.5,
    })
    downsampled_rows.append(downsampled_row)

# Convert the list of downsampled Series objects to a DataFrame
downsampled_df = pd.DataFrame(downsampled_rows)

# Combine b1 and the downsampled DataFrame
new_twins = pd.concat([b1, downsampled_df], ignore_index=True)

new_twins.to_csv(filename, index=False)

#______________________________________________________________________________________________________

filename = 'twins_model_SOL0261.csv'
pressure = pd.read_csv(filename)

# Get the unique pressure frequencies
a = pd.unique(pressure['WIND_FREQUENCY'])

print(a)

b1 = pressure[pressure['WIND_FREQUENCY'] == a[1]]
b2 = pressure[pressure['WIND_FREQUENCY'] == a[0]]

# Initialize a list to store the downsampled rows
downsampled_rows = []

# Add the first row manually with the updated frequency
first_row = b2.iloc[0].copy()
first_row['WIND_FREQUENCY'] = 0.1
downsampled_rows.append(first_row)

# Calculate the moving average for every 5th row starting from index 4
for i in range(10, len(b2), 10):
    # Create the downsampled row as a Series
    downsampled_row = pd.Series({
        'AOBT': b2.iloc[i]['AOBT'],
        'SCLK': b2.iloc[i]['SCLK'],
        'LMST': b2.iloc[i]['LMST'],
        'LTST': b2.iloc[i]['LTST'],
        'UTC': b2.iloc[i]['UTC'],
        'HORIZONTAL_WIND_SPEED': b2.iloc[i-9:i+1]['HORIZONTAL_WIND_SPEED'].mean(),
        'VERTICAL_WIND_SPEED': b2.iloc[i-9:i+1]['VERTICAL_WIND_SPEED'].mean(),
        'WIND_DIRECTION': b2.iloc[i-9:i+1]['WIND_DIRECTION'].mean(),
        'WIND_FREQUENCY': 0.1,
    })
    downsampled_rows.append(downsampled_row)

# Convert the list of downsampled Series objects to a DataFrame
downsampled_df = pd.DataFrame(downsampled_rows)

# Combine b1 and the downsampled DataFrame
new_twins = pd.concat([downsampled_df, b1], ignore_index=True)

new_twins.to_csv(filename, index=False)

#______________________________________________________________________________________________________

filename = 'twins_model_SOL0292.csv'
pressure = pd.read_csv(filename)

# Get the unique pressure frequencies
a = pd.unique(pressure['WIND_FREQUENCY'])

print(a)

b1 = pressure[pressure['WIND_FREQUENCY'] == a[0]]
b2 = pressure[pressure['WIND_FREQUENCY'] == a[1]]

# Initialize a list to store the downsampled rows
downsampled_rows = []

# Add the first row manually with the updated frequency
first_row = b2.iloc[0].copy()
first_row['WIND_FREQUENCY'] = 0.5
downsampled_rows.append(first_row)

# Calculate the moving average for every 5th row starting from index 4
for i in range(2, len(b2), 2):
    # Create the downsampled row as a Series
    downsampled_row = pd.Series({
        'AOBT': b2.iloc[i]['AOBT'],
        'SCLK': b2.iloc[i]['SCLK'],
        'LMST': b2.iloc[i]['LMST'],
        'LTST': b2.iloc[i]['LTST'],
        'UTC': b2.iloc[i]['UTC'],
        'HORIZONTAL_WIND_SPEED': b2.iloc[i-1:i+1]['HORIZONTAL_WIND_SPEED'].mean(),
        'VERTICAL_WIND_SPEED': b2.iloc[i-1:i+1]['VERTICAL_WIND_SPEED'].mean(),
        'WIND_DIRECTION': b2.iloc[i-1:i+1]['WIND_DIRECTION'].mean(),
        'WIND_FREQUENCY': 0.5,
    })
    downsampled_rows.append(downsampled_row)

# Convert the list of downsampled Series objects to a DataFrame
downsampled_df = pd.DataFrame(downsampled_rows)

# Combine b1 and the downsampled DataFrame
new_twins = pd.concat([b1, downsampled_df], ignore_index=True)

new_twins.to_csv(filename, index=False)

#______________________________________________________________________________________________________

filename = 'twins_model_SOL0475.csv'
pressure = pd.read_csv(filename)

# Get the unique pressure frequencies
a = pd.unique(pressure['WIND_FREQUENCY'])

print(a)

b1 = pressure[pressure['WIND_FREQUENCY'] == a[1]]
b2 = pressure[pressure['WIND_FREQUENCY'] == a[0]]

# Initialize a list to store the downsampled rows
downsampled_rows = []

# Add the first row manually with the updated frequency
first_row = b2.iloc[0].copy()
first_row['WIND_FREQUENCY'] = 0.1
downsampled_rows.append(first_row)

# Calculate the moving average for every 5th row starting from index 4
for i in range(10, len(b2), 10):
    # Create the downsampled row as a Series
    downsampled_row = pd.Series({
        'AOBT': b2.iloc[i]['AOBT'],
        'SCLK': b2.iloc[i]['SCLK'],
        'LMST': b2.iloc[i]['LMST'],
        'LTST': b2.iloc[i]['LTST'],
        'UTC': b2.iloc[i]['UTC'],
        'HORIZONTAL_WIND_SPEED': b2.iloc[i-9:i+1]['HORIZONTAL_WIND_SPEED'].mean(),
        'VERTICAL_WIND_SPEED': b2.iloc[i-9:i+1]['VERTICAL_WIND_SPEED'].mean(),
        'WIND_DIRECTION': b2.iloc[i-9:i+1]['WIND_DIRECTION'].mean(),
        'WIND_FREQUENCY': 0.1,
    })
    downsampled_rows.append(downsampled_row)

# Convert the list of downsampled Series objects to a DataFrame
downsampled_df = pd.DataFrame(downsampled_rows)

# Combine b1 and the downsampled DataFrame
new_twins = pd.concat([downsampled_df, b1], ignore_index=True)

new_twins.to_csv(filename, index=False)

#______________________________________________________________________________________________________