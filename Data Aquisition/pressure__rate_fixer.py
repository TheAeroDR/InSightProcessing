import pandas as pd

filename = 'ps_data_calibrated/ps_calib_SOL0168.csv' # 2 Hz or 10 Hz

pressure = pd.read_csv(filename)

# Get the unique pressure frequencies
a = pd.unique(pressure['PRESSURE_FREQUENCY'])

# some editing of this bit needs editing per file used
# Split the data based on the pressure frequency
b1 = pressure[pressure['PRESSURE_FREQUENCY'] == a[0]]
b2 = pressure[pressure['PRESSURE_FREQUENCY'] == a[1]]

# Initialize a list to store the downsampled rows
downsampled_rows = []

# Add the first row manually with the updated frequency
first_row = b2.iloc[0].copy()
first_row['PRESSURE_FREQUENCY'] = 2
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
        'PRESSURE': b2.iloc[i-4:i+1]['PRESSURE'].mean(),
        'PRESSURE_FREQUENCY': 2,
        'PRESSURE_TEMP': b2.iloc[i-4:i+1]['PRESSURE_TEMP'].mean(),
        'PRESSURE_TEMP_FREQUENCY': b2.iloc[i]['PRESSURE_TEMP_FREQUENCY']
    })
    downsampled_rows.append(downsampled_row)

# Convert the list of downsampled Series objects to a DataFrame
downsampled_df = pd.DataFrame(downsampled_rows)

# Combine b1 and the downsampled DataFrame
new_pressure = pd.concat([b1, downsampled_df], ignore_index=True)

# Save the new pressure data to a CSV file
new_pressure.to_csv(filename, index=False)


#______________________________________________________________________________________________________


filename = 'ps_data_calibrated/ps_calib_SOL0261.csv' # 10 Hz or 2 Hz

pressure = pd.read_csv(filename)

# Get the unique pressure frequencies
a = pd.unique(pressure['PRESSURE_FREQUENCY'])

# some editing of this bit needs editing per file used
# Split the data based on the pressure frequency
b2 = pressure[pressure['PRESSURE_FREQUENCY'] == a[0]]
b1 = pressure[pressure['PRESSURE_FREQUENCY'] == a[1]]

# Initialize a list to store the downsampled rows
downsampled_rows = []

## Add the first row manually with the updated frequency
first_row = b2.iloc[0].copy()
first_row['PRESSURE_FREQUENCY'] = 2
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
        'PRESSURE': b2.iloc[i-4:i+1]['PRESSURE'].mean(),
        'PRESSURE_FREQUENCY': 2,
        'PRESSURE_TEMP': b2.iloc[i-4:i+1]['PRESSURE_TEMP'].mean(),
        'PRESSURE_TEMP_FREQUENCY': b2.iloc[i]['PRESSURE_TEMP_FREQUENCY']
    })
    downsampled_rows.append(downsampled_row)

# Convert the list of downsampled Series objects to a DataFrame
downsampled_df = pd.DataFrame(downsampled_rows)

# Combine b1 and the downsampled DataFrame
new_pressure = pd.concat([b1, downsampled_df], ignore_index=True)

# Save the new pressure data to a CSV file
new_pressure.to_csv(filename, index=False)

#______________________________________________________________________________________________________

filename = 'ps_data_calibrated/ps_calib_SOL0612.csv' # 10 Hz NaN Hz (region of time where temp sampled at 20 Hz and pressure at 10) or 20 Hz

pressure = pd.read_csv(filename)

# Get the unique pressure frequencies
a = pd.unique(pressure['PRESSURE_FREQUENCY'])

# some editing of this bit needs editing per file used
# Split the data based on the pressure frequency
b1 = pressure[pressure['PRESSURE_FREQUENCY'] == a[0]]
b2 = pressure[pressure['PRESSURE_FREQUENCY'] == a[2]]

# Initialize a list to store the downsampled rows
downsampled_rows = []

# Add the first row manually with the updated frequency
#first_row = b2.iloc[1].copy()
#first_row['PRESSURE_FREQUENCY'] = 10
#downsampled_rows.append(first_row)

# Calculate the moving average for every 5th row starting from index 4
for i in range(1, len(b2), 2):
    # Create the downsampled row as a Series
    downsampled_row = pd.Series({
        'AOBT': b2.iloc[i]['AOBT'],
        'SCLK': b2.iloc[i]['SCLK'],
        'LMST': b2.iloc[i]['LMST'],
        'LTST': b2.iloc[i]['LTST'],
        'UTC': b2.iloc[i]['UTC'],
        'PRESSURE': b2.iloc[i-1:i+1]['PRESSURE'].mean(),
        'PRESSURE_FREQUENCY': 10,
        'PRESSURE_TEMP': b2.iloc[i-1:i+1]['PRESSURE_TEMP'].mean(),
        'PRESSURE_TEMP_FREQUENCY': b2.iloc[i]['PRESSURE_TEMP_FREQUENCY']
    })
    downsampled_rows.append(downsampled_row)

# Convert the list of downsampled Series objects to a DataFrame
downsampled_df = pd.DataFrame(downsampled_rows)

# Combine b1 and the downsampled DataFrame
new_pressure = pd.concat([b1, downsampled_df], ignore_index=True)

# Save the new pressure data to a CSV file
new_pressure.to_csv(filename, index=False)
