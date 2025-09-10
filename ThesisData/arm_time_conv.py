from planetary_coverage import MetaKernel
import spiceypy as spice
from obspy import UTCDateTime
import os
import numpy as np
import pandas as pd

def batch_utc2ltst(utc_times):
    """Convert multiple UTC times to LTST in a vectorized way"""
    # Convert all times to ET at once
    et_values = np.array([spice.utc2et(t.strftime("%Y-%m-%dT%H:%M:%S")) for t in utc_times.flatten()])
    
    # Calculate sols for all times at once using vectorized operations
    insight_longitude = 135.623 * spice.rpd()
    
    # Get landing reference time
    landing_utc = UTCDateTime("2018-11-26T19:52:59")
    landing_et = spice.utc2et(landing_utc.strftime("%Y-%m-%dT%H:%M:%S"))
    
    # Get sol0 start (midnight before landing)
    landing_lst = spice.et2lst(landing_et, 499, insight_longitude, "PLANETOCENTRIC", 3, 24)
    time_since_midnight = landing_lst[0] * 3600 + landing_lst[1] * 60 + landing_lst[2]
    sol0_start_et = landing_et - time_since_midnight
    
    # Calculate sol numbers using vectorized math
    sols = np.floor((et_values - sol0_start_et) / 88775.244).astype(int)
    
    # Get LTST for each time (this still needs individual calls)
    ltst_values = []
    for i, et in enumerate(et_values):
        ltst = spice.et2lst(et, 499, insight_longitude, "PLANETOCENTRIC", 3, 24)
        ltst_str = f"{sols[i]:05d} {int(ltst[0]):02d}:{int(ltst[1]):02d}:{int(ltst[2]):02d}"
        ltst_values.append(ltst_str)
    
    return np.reshape(ltst_values, [-1, 1])

arm_files = ['./arm_activity.txt']

mk = MetaKernel('../Data Aquisition/spice_kernels/mk/insight_v16.tm', kernels='../Data Aquisition/spice_kernels')

output_file = './arm_activity_LTST.txt'
spice.furnsh(mk)

for f in arm_files:
    print(f"Processing {f}...")

    filepath = os.path.join(f)
    
    df = pd.read_csv(filepath,header=None)
    
    a = df.iloc[:, 0].values
    ltst_values = batch_utc2ltst(np.array([UTCDateTime(t) for t in a]))
    
    df.insert(1,"LTST Start", ltst_values)

    a = df.iloc[:, 2].values
    ltst_values = batch_utc2ltst(np.array([UTCDateTime(t) for t in a]))
    
    df.insert(3,"LTST Stop", ltst_values)
    
    output_path = os.path.join(output_file) 
    with open(output_path, 'w') as file: 
        df.to_csv(file, index=False, header=False)
    
    print(f"Saved to {output_path}")

print("All files processed successfully!")

spice.unload(mk)

