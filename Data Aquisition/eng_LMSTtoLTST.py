from planetary_coverage import MetaKernel
import spiceypy as spice
from obspy import UTCDateTime
import os
import numpy as np
import pandas as pd


eng_path = './sc_eng_data/'
eng_files = [f for f in os.listdir(eng_path) if f.startswith('ancil_SOL')]

mk = MetaKernel('./seis_data_calibrated_LTST/spice_kernels/mk/insight_v16.tm', kernels='seis_data_calibrated_LTST/spice_kernels')

output_dir = './sc_eng_data_LTST/'
spice.furnsh(mk)

for f in eng_files:
    print(f"Processing {f}...")

    filepath = os.path.join(eng_path, f)
    
    with open(filepath, 'r') as file:
        header = file.readline().strip()

    header_columns = header.split(',')
    header_columns[3] = 'LTST'
    new_header = ','.join(header_columns)

    df = pd.read_csv(filepath, skiprows=1)
    
    a = df.iloc[:, 1].values
    UTC = np.array([UTCDateTime(utc) for utc in a])
    
    et_values = np.array([spice.utc2et(t.strftime("%Y-%m-%dT%H:%M:%S")) for t in UTC.flatten()])
    insight_longitude = 135.623 * spice.rpd()     
    ltst_values = []
    for i, et in enumerate(et_values):
        ltst = spice.et2lst(et, 499, insight_longitude, "PLANETOCENTRIC", 3, 24)
        ltst_str = f"{int(ltst[0]):02d}:{int(ltst[1]):02d}:{int(ltst[2]):02d}"
        ltst_values.append(ltst_str)
    
    df.iloc[:, 3] = ltst_values
    
    output_path = os.path.join(output_dir, f)
    
    with open(output_path, 'w') as file:
        file.write(new_header + '\n')  
        df.to_csv(file, index=False, header=False)
    
    print(f"Saved to {output_path}")

print("All files processed successfully!")

spice.unload(mk)
