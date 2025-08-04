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

def sol_day_to_utc(Sol):
    sol0=obspy.UTCDateTime(2018, 11, 26, 5, 10, 50.33508)
    sec_per_day_mars=88775.2440
    sec_per_day_earth=86400
    start_SET = (Sol * sec_per_day_mars)
    end_SET = ((Sol + 1) * sec_per_day_mars)
    start_date = obspy.UTCDateTime(float(sol0) + start_SET)
    end_date = obspy.UTCDateTime(float(sol0) + end_SET)
    return start_date, end_date

def utc_to_sol(utc):
    sol0=obspy.UTCDateTime(2018, 11, 26, 5, 10, 50.33508)
    sec_per_day_mars=88775.2440
    sec_per_day_earth=86400
    second_time = (float(utc))-float(sol0)
    sol = second_time / sec_per_day_mars
    return sol

def sol_to_utc(sol):
    sol0=obspy.UTCDateTime(2018, 11, 26, 5, 10, 50.33508)
    sec_per_day_mars=88775.2440
    sec_per_day_earth=86400
    second_time = (sol * sec_per_day_mars)+float(sol0)
    utc = obspy.UTCDateTime(second_time)
    return utc

def get_data(sol):
    print(f"Sol {sol}")
    start_date, end_date = sol_day_to_utc(sol)
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
        sol_times = np.reshape([np.array([utc_to_sol(t) for t in utc_times])],[-1,1])
        data = np.concatenate([utc_times, sol_times, data],1)
        save_seis(sol,data,rate)
        print(f"Sol {sol} complete")

    except:
        print(f'no data for Sol: {sol}')

def save_seis(sol, data, rates):
    filename = f"seis_cal_SOL{str(sol).zfill(4)}_{int(rates)}Hz_v1.csv"
    df = pd.DataFrame(data,columns = ['SCET_UTC ', 'MLST', 'VBB_N', 'VBB_E', 'VBB_Z'])
    df.to_csv(filename, index = False)

if __name__ == '__main__':
    with mp.Pool(processes=4) as pool: # set the number of processes to run in parallel
        result = pool.starmap_async(get_data, [(i,) for i in range(560, 1250)])
        result.get()
        pool.close()
        pool.join()