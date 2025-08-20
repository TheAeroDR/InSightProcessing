import obspy
from obspy.clients.fdsn import Client
from obspy.signal.rotate import rotate2zne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import pandas as pd
import multiprocessing as mp
import warnings
from planetary_coverage import MetaKernel
import spiceypy as spice

#warnings.filterwarnings('ignore')


mpl.rcParams['axes.formatter.useoffset'] = False

def remove(st, starttime, endtime, loc, chan):
    #Function to remove the instrument response
    st_raw = st.copy()
    client = obspy.clients.fdsn.Client("IRIS")
    inv = client.get_stations(network="XB",
                              station="ELYSE",
                              location=loc,
                              channel=chan,
                              starttime=starttime,
                              endtime=endtime,
                              level='response')
    st_raw.detrend(type='demean')
    st_raw.taper(0.1)
    st_raw.detrend()
    
    for tr in st_raw:
        try:
            tr.remove_response(inv, pre_filt=(0.01, 0.02, 50, 60),
                                output='VEL')
        except ValueError:
            print('Could not remove response for channel:')
            print(tr)
    return st_raw

def rotate(u, v, w, starttime, endtime, loc, chanx, chany, chanz):         
    #function to rotate to ZNE
    client = obspy.clients.fdsn.Client("IRIS")

    u = u[0].data
    u_inv = client.get_stations(network="XB", station="ELYSE", location=loc, channel=chanx, starttime=starttime, endtime=endtime, level='response')
    u_azi = u_inv[0][0][0].azimuth
    u_dip = u_inv[0][0][0].dip

    v = v[0].data
    v_inv = client.get_stations(network="XB", station="ELYSE", location=loc, channel=chany, starttime=starttime, endtime=endtime, level='response')
    v_azi = v_inv[0][0][0].azimuth
    v_dip = v_inv[0][0][0].dip

    w = w[0].data
    w_inv = client.get_stations(network="XB", station="ELYSE", location=loc, channel=chanz, starttime=starttime, endtime=endtime, level='response')
    w_azi = w_inv[0][0][0].azimuth
    w_dip = w_inv[0][0][0].dip

    Z = (u * math.sin(math.radians(u_dip))) + (v * math.sin(math.radians(v_dip))) + (w * math.sin(math.radians(w_dip)))
    N = (u * math.cos(math.radians(u_dip)) * math.cos(math.radians(u_azi))) + (v * math.cos(math.radians(v_dip)) * math.cos(math.radians(v_azi))) + (w * math.cos(math.radians(w_dip)) * math.cos(math.radians(w_azi)))
    E = (u * math.cos(math.radians(u_dip)) * math.sin(math.radians(u_azi))) + (v * math.cos(math.radians(v_dip)) * math.sin(math.radians(v_azi))) + (w * math.cos(math.radians(w_dip)) * math.sin(math.radians(w_azi)))
    return np.concatenate([np.reshape(N,[-1,1]), np.reshape(E,[-1,1]), np.reshape(Z,[-1,1])], axis = 1)

def utc2ltst(utc_time):
    spice_utc = utc_time.strftime("%Y-%m-%dT%H:%M:%S")
    et = spice.utc2et(spice_utc)
    insight_longitude = 135.623 * spice.rpd()  # Convert to radians
    ltst = spice.et2lst(et, 499, insight_longitude, "PLANETOCENTRIC", 3, 24)
    sol = ltstsol(et,insight_longitude)
    ltst_str = f"{sol:05d} {int(ltst[0]):02d}:{int(ltst[1]):02d}:{int(ltst[2]):02d}"


    return ltst_str

def ltstsol(et, insight_longitude=135.623 * spice.rpd()):
 
    landing_utc = obspy.UTCDateTime("2018-11-26T19:52:59")

    landing_et = spice.utc2et(landing_utc.strftime("%Y-%m-%dT%H:%M:%S"))

    landing_lst = spice.et2lst(landing_et, 499, insight_longitude, "PLANETOCENTRIC", 3, 24)

    start_sol0 = et_preceeding_midnight(landing_et - (landing_lst[0] * 3600 + landing_lst[1] * 60 + landing_lst[2]), insight_longitude)

        # Calculate the mean sol number

    mean_sol = int((et - start_sol0)/88775.244)

        # Find exact ET of midnight for this sol
    this_sol_midnight_et = et_neartest_midnight(start_sol0 + (mean_sol * 88775.244))

    # If target time is before this sol's start, adjust sol number
    if et < this_sol_midnight_et:
        return mean_sol - 1
    
    # If target time is after this sol's start but before next sol's start, this is correct sol
    next_sol_midnight_et = et_neartest_midnight(this_sol_midnight_et + 88775.244)
    if et < next_sol_midnight_et:
        return mean_sol

    return mean_sol

def et_neartest_midnight(et_estimate,insight_longitude=135.623 * spice.rpd()):
    max_iterations = 10
    tolerance = 0.1  # seconds
    et = et_estimate
    
    for i in range(max_iterations):
        ltst = spice.et2lst(et, 499, insight_longitude, "PLANETOCENTRIC", 3, 24)
        h = int(ltst[0])
        m = int(ltst[1])
        s = int(ltst[2])
    
        if h < 12:
            seconds_from_midnight = (h * 3600 + m * 60 + s)
        else:
            seconds_from_midnight = (h - 24) * 3600 + m * 60 + s
        if abs(seconds_from_midnight) < tolerance:
            break
        
        damping = 0.7
        et -= seconds_from_midnight * damping
    
    return et

def et_preceeding_midnight(et_estimate,insight_longitude=135.623 * spice.rpd()):
    max_iterations = 10
    tolerance = 0.1  # seconds
    et = et_estimate
    
    for i in range(max_iterations):
        ltst = spice.et2lst(et, 499, insight_longitude, "PLANETOCENTRIC", 3, 24)
        h = int(ltst[0])
        m = int(ltst[1])
        s = int(ltst[2])
    
        seconds_from_midnight = (h * 3600 + m * 60 + s)
        
        if abs(seconds_from_midnight) < tolerance:
            break
        
        damping = 0.7
        et -= seconds_from_midnight * damping
    
    return et
def batch_utc2ltst(utc_times):
    """Convert multiple UTC times to LTST in a vectorized way"""
    # Convert all times to ET at once
    et_values = np.array([spice.utc2et(t.strftime("%Y-%m-%dT%H:%M:%S")) for t in utc_times.flatten()])
    
    # Calculate sols for all times at once using vectorized operations
    insight_longitude = 135.623 * spice.rpd()
    
    # Get landing reference time
    landing_utc = obspy.UTCDateTime("2018-11-26T19:52:59")
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
    
def et2lmst(et):
    # predicted lander location
    insight_longitude =  135.97 * spice.rpd()
      
    days_since_j2000 = (et / 86400.0)
    
    # Calculate Allison-McEwen parameters as https://www.giss.nasa.gov/tools/mars24/help/algorithm.html
    M = 19.3871 + 0.52402073 * days_since_j2000
    M_rad = math.radians(M % 360.0)
    
    a_fms = 270.3871 + 0.524038496 * days_since_j2000
    a_fms = a_fms % 360.0
    
    eq_centre = (10.691 + days_since_j2000 * 3e-7) * math.sin(M_rad) + \
                    0.623 * math.sin(2 * M_rad) + \
                    0.050 * math.sin(3 * M_rad) + \
                    0.005 * math.sin(4 * M_rad) + \
                    0.0005 * math.sin(5 * M_rad)
    
    PBS = 0.0
    cycles = [(2.2353, 49.409, 0.0071), (2.7543, 168.173, 0.00573), 
              (1.1177, 191.837, 0.0039), (15.7866, 21.739, 0.0037),
              (2.1354, 15.704, 0.0021), (2.4694, 95.528, 0.0020), 
              (32.8493, 49.095, 0.0018)]
              
    for period, phase, amp in cycles:
        PBS += amp * math.cos(math.radians(0.985626 * days_since_j2000 / period + phase))
    
    eq_centre += PBS
    
    Ls = a_fms + eq_centre
    Ls_rad = math.radians(Ls % 360.0)
    
    eot = 2.861 * math.sin(2.0 * Ls_rad) - \
          0.071 * math.sin(4.0 * Ls_rad) + \
          0.002 * math.sin(6.0 * Ls_rad) - \
          eq_centre
    
    eot_hours = eot / 15.0
    
    ltst = spice.et2lst(et, 499, insight_longitude, "PLANETOCENTRIC", 3, 24)
    
    ltst_hours = ltst[0] + ltst[1] / 60.0 + ltst[2] / 3600.0
    
    # Calculate LMST by applying equation of time correction
    lmst_hours = (ltst_hours - eot_hours) % 24.0
    
    lmst_hours_int = int(lmst_hours)
    lmst_mins = int((lmst_hours - lmst_hours_int) * 60.0)
    lmst_secs = ((lmst_hours - lmst_hours_int) * 60.0 - lmst_mins) * 60.0
        
    # Format as HH:MM:SS
    lmst_str = f"{lmst_hours_int:02d}:{lmst_mins:02d}:{lmst_secs:06.3f}"

    return lmst_str

def sol2utcrange(sol_number):
    landing_utc = obspy.UTCDateTime("2018-11-26T19:52:59")

    landing_et = spice.utc2et(landing_utc.strftime("%Y-%m-%dT%H:%M:%S"))
    landing_lmst = et2lmst(landing_et)
    landing_lmst_parts = landing_lmst.split(":")
    landing_lmst_hours = int(landing_lmst_parts[0])
    landing_lmst_mins = int(landing_lmst_parts[1])
    landing_lmst_secs = float(landing_lmst_parts[2])
    
    time_since_midnight = landing_lmst_hours * 3600 + landing_lmst_mins * 60 + landing_lmst_secs
    
    sol0_start_et_estimate = landing_et - time_since_midnight

    sol0_start_et = et_preceeding_midnight_lmst(sol0_start_et_estimate)

    sol_start_et = (sol0_start_et + (sol_number * 88775.244))

    sol_end_et = (sol_start_et + 88775.244)
    
    # Convert ET back to UTC
    sol_start_utc = spice.et2utc(sol_start_et, "ISOC", 3)
    sol_end_utc = spice.et2utc(sol_end_et, "ISOC", 3)

    return [obspy.UTCDateTime(sol_start_utc), obspy.UTCDateTime(sol_end_utc)]

def et_preceeding_midnight_lmst(et_estimate):
    max_iterations = 1000
    tolerance = 0.1  # seconds
    et = et_estimate
    
    for i in range(max_iterations):
        lmst = et2lmst(et)
        lmst_parts = lmst.split(":")
        h = int(lmst_parts[0])
        m = int(lmst_parts[1])
        s = float(lmst_parts[2])

        if h < 12:
            seconds_from_midnight = (h * 3600 + m * 60 + s)
        else:
            seconds_from_midnight = ((h - 24) * 3600 + m * 60 + s)

        if abs(seconds_from_midnight) < tolerance:
            break
        
        damping = 0.7
        et -= seconds_from_midnight * damping
    
    return et

def load_spice_kernels():
    mk = MetaKernel('seis_data_calibrated_LTST/spice_kernels/mk/insight_v16.tm', kernels='seis_data_calibrated_LTST/spice_kernels')

    spice.furnsh(mk)

def get_data(sol):
    print(f"Sol {sol}")
    start_date, end_date = sol2utcrange(sol)
    client = Client("IPGP")
    network ='XB'
    station = 'ELYSE'
    location = '0?'
    channel_u = 'BHU'
    channel_v = 'BHV'
    channel_w = 'BHW'

    try:
        stream_u = client.get_waveforms(network, station, location, channel_u, start_date, end_date)
        stream_u = remove(stream_u, start_date, end_date, location, channel_u)

        stream_v = client.get_waveforms(network, station, location, channel_v, start_date, end_date)
        stream_v = remove(stream_v, start_date, end_date, location, channel_v)
        
        stream_w = client.get_waveforms(network, station, location, channel_w, start_date, end_date)
        stream_w = remove(stream_w, start_date, end_date, location, channel_w)
        
        
        data = rotate(stream_u, stream_v, stream_w, start_date, end_date, location, channel_u, channel_v, channel_w)
        
        rate = stream_u[0].stats.sampling_rate

        timevals = stream_u[0].times() + float(stream_u[0].stats.starttime)
        utc_times = np.reshape([obspy.UTCDateTime(t) for t in timevals],[-1,1])
        sol_times = batch_utc2ltst(utc_times)    
        data = np.concatenate([utc_times, sol_times, data],1)
        save_seis(sol,data,rate)
        print(f"Sol {sol} complete")

    except:
        print(f'no data for Sol: {sol}')

def save_seis(sol, data, rates):
    filename = f"seis_cal_SOL{str(sol).zfill(4)}_{int(rates)}Hz_v2.csv"
    df = pd.DataFrame(data,columns = ['SCET_UTC ', 'LTST', 'VBB_N', 'VBB_E', 'VBB_Z'])
    df.to_csv(filename, index = False)

if __name__ == '__main__':
    load_spice_kernels()
    
    with mp.Pool(processes=4) as pool: # set the number of processes to run in parallel
        result = pool.starmap_async(get_data, [(i,) for i in range(75, 1250)])
        result.get()
        pool.close()
        pool.join()
    
    spice.kclear()