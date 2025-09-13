#import necessary modules
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import warnings
import logging
warnings.filterwarnings('ignore')

# Define a function to get the file paths for magnetic, pressure, twins, and engineering data for a given Sol number
def SolPath(Sol):
    """
    Returns file paths for magnetic, pressure, twins, and engineering data for a given Sol number.

    Args:
        Sol (int): The Sol number.

    Returns:
        list: A list of file paths for magnetic, pressure, twins, and engineering data.
    """
    # Set the file paths to the directories where the data is stored
    ifg_path = './ifg_data_calibrated/'
    ps_path = './ps_data_calibrated/'
    #twins_path = './twins_data_calibrated/'
    twins_path = './twins_data_derived/'
    eng_path = './sc_eng_data/'
    seis_path = './seis_data_calibrated/'

    # Filter the files in each directory based on the Sol number
    ifg_files = [f for f in os.listdir(ifg_path) if f.startswith('ifg_cal_SOL'+str(Sol).zfill(4))]
    
    ifg_files = mag_file_pick(ifg_files, ifg_path)
    ps_files = [f for f in os.listdir(ps_path) if f.startswith('ps_calib_SOL'+str(Sol).zfill(4))]
    #twins_files = [f for f in os.listdir(twins_path) if f.startswith('twins_calib_SOL'+str(Sol).zfill(4))]
    twins_files = [f for f in os.listdir(twins_path) if f.startswith('twins_model_SOL'+str(Sol).zfill(4))]
    eng_files = [f for f in os.listdir(eng_path) if f.startswith('ancil_SOL'+str(Sol).zfill(4))]
    seis_files = [f for f in os.listdir(seis_path) if f.startswith('seis_cal_SOL'+str(Sol).zfill(4))]

    logging.info(f"SOL {Sol}: {ifg_files},{ps_files},{twins_files},{eng_files},{seis_files}")

    # If all files exist return paths
    if ifg_files and ps_files and twins_files and eng_files and seis_files:

        # Join the directory paths with the filtered filenames and create a list of file paths
        paths = [
            os.path.join(ifg_path, next(iter(ifg_files), None)),
            os.path.join(ps_path, next(iter(ps_files), None)),
            os.path.join(twins_path, next(iter(twins_files), None)),
            os.path.join(eng_path, next(iter(eng_files), None)),
            os.path.join(seis_path, next(iter(seis_files), None))
        ]
    
    # Else return an empty list
    else:
        paths = []

    # Return the list of file paths
    return paths

# Define function to select magnetic file based on highest rate for full day
def mag_file_pick(ifg_files, ifg_path):

    # Create data frame
    mag_list = pd.DataFrame(columns=['file','rate','duration'])
    magnetic_rate_dict = {'pt2Hz': 0.2, '2Hz': 2.0, '20Hz': 20.0, '10Hz': 10.0, 'gpt2Hz': 0.2}

    # Loop through all ifg files to store file name, duration of dataset contained, sensing rate
    for i, file in enumerate(ifg_files):
        time = np.loadtxt(os.path.join(ifg_path, file), skiprows = 1, usecols = [1], dtype=float)
        duration = time[-1] - time[0]
        mag_rate = file.split('_')[3]
        mag_rate = magnetic_rate_dict.get(mag_rate, None)
        mag_list.loc[i] = {'file':file, 'rate':mag_rate,'duration':duration}
    
    # Keep entries with recording longer than 0.2 days
    mag_list = mag_list[mag_list.duration > 0.2]

    #reject 20 Hz data set - too long to process
    mag_list = mag_list[mag_list.rate != 20.0]

    # Sort by descending rate to put the highest rate at the top
    mag_list = mag_list.sort_values('rate',ascending=False)
    #print(mag_list)
    if mag_list.empty:
        return []
    # Return 'best' data set
    return [mag_list.file.iloc[0]]

# Define a function to extract magnetic, pressure, twins, seismic, and engineering data for a given file path
def data_extract(temp):
    """
    Extracts magnetic, pressure, twins, seismic, and engineering data for a given Sol number.

    Args:
        temp (list): list of file paths for the ifg, ps, twins and eng files.

    Returns:
        tuple: A tuple containing magnetic, magnetic_rate, pressure, pressure_rate, temperature, twins, twins rate, seismic, seismic rate and selected engingeering data.
    """
    # Load magnetic data
    magnetic = np.loadtxt(temp[0], skiprows = 1, usecols = [1, 7, 8, 9], dtype=float)

    # Get the sampling rate for the magnetic data
    rate_dict = {'pt2Hz': 0.2, '2Hz': 2.0, '20Hz': 20.0, '10Hz': 10.0, 'gpt2Hz': 0.2, '100Hz': 100.0}
    magnetic_rate_str = temp[0].split('_')[5]
    magnetic_rate = rate_dict.get(magnetic_rate_str, None)

    # Load pressure datayou
    pressure = np.loadtxt(temp[1], delimiter=',', skiprows = 1, usecols=[2, 5], dtype=float, converters={
        2: time2decimal, 
        5: lambda s: float(s or np.nan)
    })
    # Load pressure rate data and unifiy it to one number as the average
    pressure_rate = np.nanmean(np.loadtxt(temp[1], delimiter=b',', skiprows = 1, usecols=[6], dtype=float, converters={
        6: lambda s: float(s or np.nan)
    }))

    # Load twins data
    #twins = np.loadtxt(temp[2], delimiter=',', skiprows = 1, usecols=[2, 5, 6, 7, 15, 16, 17], dtype=float, converters={
    #    2: time2decimal,
    #    5: lambda s: float(s or np.nan),
    #    6: lambda s: float(s or np.nan),
    #    7: lambda s: float(s or np.nan),
    #    15: lambda s: float(s or np.nan),
    #    16: lambda s: float(s or np.nan),
    #    17: lambda s: float(s or np.nan)
    #})

    # Calculate twins rate similarly to the pressure rate
    #twins_rate = np.nanmean(np.loadtxt(temp[2], delimiter=',', skiprows = 1, usecols=[8], dtype=float, converters={
    #    8: lambda s: float(s or np.nan)
    #}))

    twins = np.loadtxt(temp[2], delimiter=',', skiprows = 1, usecols=[2, 5, 6, 7, 10, 13], dtype=float, converters={
        2: time2decimal,
        5: lambda s: float(s or np.nan),
        6: lambda s: float(s or np.nan),
        7: lambda s: float(s or np.nan),
        10: lambda s: float(s or np.nan),
        13: lambda s: float(s or np.nan)
    })

    twins_rate = np.nanmean(np.loadtxt(temp[2], delimiter=',', skiprows = 1, usecols=[8], dtype=float, converters={
        8: lambda s: float(s or np.nan)
    }))

    # From the twins data, temperature column, extract all the exisiting values (i.e. not NaN) as the temperature data, and average the two booms
    temperatureM = twins[~np.isnan(twins[:,4]), :][:,[0,4]]
    temperatureP = twins[~np.isnan(twins[:,5]), :][:,[0,5]]

    temperature = temperatureM

    twins = np.delete(twins, np.where(np.isnan(twins[:, [1, 3]]).any(axis=1))[0], axis = 0)

    # Remove the temperature columns from the twins data
    twins = twins[:,[0,1,2,3]]

    # load seismic data
    seis_data = np.loadtxt(temp[4], delimiter=',', skiprows = 1, usecols=[1, 2, 3, 4], dtype=float, converters={
        1: lambda s: float(s or np.nan),
        2: lambda s: float(s or np.nan),
        3: lambda s: float(s or np.nan),
        4: lambda s: float(s or np.nan)
    })
    
    seis_rate_str = temp[4].split('_')[5]
    seis_rate = rate_dict.get(seis_rate_str, None)

    # Load engineering data
    eng_data = np.loadtxt(temp[3], delimiter=',', skiprows=1, usecols=[3,45,48,49,50], dtype=float, converters={
        3: lambda s: time2decimal(s[4:-1]),
        45: lambda s: float(s) if s and float(s) < 9999 else np.nan,
        48: lambda s: float(s) if s and float(s) < 9999 else np.nan,
        49: lambda s: float(s) if s and float(s) < 9999 else np.nan,
        50: lambda s: float(s) if s and float(s) < 9999 else np.nan
    })

    # Return a tuple containing magnetic, magnetic_rate, pressure, pressure_rate, temperature, twins, twins_rate, and eng_data
    return magnetic, magnetic_rate, pressure, pressure_rate, temperature, twins, twins_rate, seis_data, seis_rate, eng_data

# Define a function to convert a time format to decimal time
def time2decimal(time):
    """
    Convert time in the format "DayOfYearM:hour:minute:second" to decimal time format.

    Parameters:
    time (pandas.Series): Time in the format "DayOfYearM:hour:minute:second".

    Returns:
    decimaltime (numpy.ndarray): Decimal time array.
    """
    def convert_time(time_str):
        """
        Convert a single time string to decimal time format.

        Args:
            time_str (str): A time string in the format "DayOfYearM:hour:minute:second".

        Returns:
            float: Decimal time value.
        """

        if isinstance(time_str, str):
            time_str = time_str.encode('utf-8')

        # Split string at the 'M', will give an integer day number and a string of the time
        dayofyear, time_str = time_str.split(b'M', 1)

        # Split string at the ':' values into integers of hour and minue, and a float of seconds
        hour, minute, second = time_str.split(b':')
        
        # Combine the integer day with decimalised hour, minite and second values
        decimaltime = int(dayofyear) + (int(hour) / 24.0) + (int(minute) / 1440.0) + (float(second) / 88775.2440)
        decimaltime = int(dayofyear) + (int(hour) / 24.0) + (int(minute) / 1440.0) + (float(second) / 86400.0)
        
        return decimaltime
    
    #call the inner function    
    decimaltime = convert_time(time)
    return decimaltime

# Define a function to detrend data
def detrend(data, rate, window):
    """
    Calculate detrend the input data by time averaging and removing the mean.

    Parameters:
    data (numpy.ndarray): Input data in the shape of (n_samples, n_features).
    rate (float): Sampling rate of the data.
    window (float): Window size for the time averaging in seconds.

    Returns:
    output (numpy.ndarray): Detrended data with the same shape as the input data.
    background (numpy.ndarray): Background signal of the same shape as the input data.
    """

    # Extract time values and data values from input array
    time_values = data[:, 0]
    data_values = data[:, 1:]

    # Calculate the number of samples in the data
    n_samples = data_values.shape[0]

    # Calculate the number of samples within the sliding window
    window_size = int(window * rate)

    # Initialize arrays for detrended data and background signal
    detrended_data = np.zeros_like(data_values)
    background = np.zeros_like(data_values)

    # Loop through all columns (features) in the data
    for i in range(data_values.shape[1]):
        # Loop through the time values and calculate the average within a sliding window
        for j in range(n_samples - window_size):
            window_values = data_values[j : j + window_size, i]
            window_mean = np.nanmean(window_values)
            detrended_data[j + window_size // 2, i] = data_values[j + window_size // 2, i] - window_mean
            background[j + window_size // 2, i] = window_mean

    # Concatenate the detrended data with the time values
    output = np.column_stack((time_values, detrended_data))

    return output, background

# Define a function to time average data
def time_average(data, rate, window):
    """
    Calculate the average within a sliding window.

    Parameters:
    data (numpy.ndarray): Input data in the shape of (n_samples, n_features).
    rate (float): Sampling rate of the data.
    window (float): Window size for the time averaging in seconds.

    Returns:
    averaged_data (numpy.ndarray): Averaged data with the time values included.
    """

    # Extract time values and data values from input array
    time_values = data[:, 0]
    data_values = data[:, 1:]

    # Calculate the number of samples in the data
    n_samples = data_values.shape[0]

    # Calculate the number of samples within the sliding window
    window_size = int(window * rate)

    # Initialize an array for averaged data
    averaged_data = np.zeros_like(data_values)

    # Loop through all columns (features) in the data
    for i in range(data_values.shape[1]):
        # Loop through the time values and calculate the average within a sliding window
        for j in range(n_samples):
            start_index = max(0, j - window_size // 2)
            end_index = min(n_samples, j + window_size // 2 + 1)
            window_values = data_values[start_index:end_index, i]
            averaged_data[j, i] = np.nanmean(window_values)

    # Concatenate the averaged data with the corresponding time values
    output = np.column_stack((time_values, averaged_data))

    return output

# Define a function to downsample and filter high rate data
def downsample(data, sample_rate, downsample_ratio):
    # Define anti-aliasing FIR filters for each downsampling ratio
    normalised_co = 39.8 / 100
    filter_coeffs = {
        2: sp.signal.firwin(301, normalised_co / 2),
        4: sp.signal.firwin(301, normalised_co / 4),
        5: np.load('FIR_coeffs_5.npy').tolist()
        }

    time_column = data[:, 0]
    decimated_time = time_column[::downsample_ratio]
    decimated_data = []

    for channel in data[:, 1:].T:
        # Apply the appropriate FIR filter based on the downsampling ratio
        filtered_channel = sp.signal.convolve(channel, filter_coeffs[downsample_ratio], mode='same')

        # Decimate the filtered data with the corresponding ratio
        decimated_channel = filtered_channel[::downsample_ratio]

        decimated_data.append(decimated_channel)
    
    
    # Calculate the new sample rate
    new_sample_rate = sample_rate / downsample_ratio
    a = np.column_stack((decimated_time, *decimated_data))
    return a, new_sample_rate

# Define a function to low pass pressure data to 2Hz
def lowpass_2Hz(data, sample_rate):
    """
    Apply a low-pass Butterworth filter with a 2 Hz cutoff frequency to the data.
    
    Parameters:
    - data (numpy array): The input signal (e.g., pressure data).
    - sample_rate (float): The sampling rate of the data in Hz.
    
    Returns:
    - data_out (numpy array): The filtered signal.
    - sample_rate (float): The new sample rate, which doesn't change in the filtering process.
    """
    if sample_rate == 2:
        return data, sample_rate
    else:
        cutoff_frequency = 2.0  # Hz
        
        nyquist_frequency = sample_rate / 2.0
        normalized_cutoff = cutoff_frequency / nyquist_frequency

        b, a = sp.signal.butter(N=2, Wn=normalized_cutoff, btype='low', analog=False)

        filtered_data = sp.signal.filtfilt(b, a, data[:,1])

        data_out = np.column_stack([data[:,0],filtered_data])
        return data_out, sample_rate

# Define a function to differentiate a dataset
def differentiate(data, sample_rate):
    time = data[:,0]
    velocities = data[:, 1:]
    
    # Calculate acceleration components
    dt = 1 / sample_rate
    accelerations = np.gradient(velocities, dt, axis=0)
    
    # Combine time and acceleration components into a new array
    acceleration = np.column_stack((time, accelerations))
    
    return acceleration

# Define a function to apply a band pass filter
def bandpass(data, sample_rate, low_freq, high_freq):
    time = data[:, 0]
    channels = data[:, 1:]
    filtered_channels = []
    nyquist_freq = sample_rate / 2.0

    if high_freq > nyquist_freq:
        logging.warning(f"Warning: High frequency ({high_freq}) exceeds Nyquist frequency ({nyquist_freq}). Adjusting high frequency to Nyquist frequency.")
        high_freq = nyquist_freq

    for channel in channels.T:
        # Apply the bandpass filter to the channel
        b, a = sp.signal.butter(4, [low_freq, high_freq], fs=sample_rate, btype='band')
        filtered_channel = sp.signal.filtfilt(b, a, channel)
        filtered_channels.append(filtered_channel)

    filtered_data = np.column_stack((time, np.array(filtered_channels).T))
    return filtered_data

# Define a function to find the valleys in a detrened pressure signal
def find_signal_peaks(signal, threshold, peak_distance):
    """
    Finds peaks in a pressure signal and returns their indices.
    
    Args:
        signal (numpy array): the signal to find peaks in. 
            The first column is assumed to contain the time values, 
            and the second column is assumed to contain the signal values.
        threshold (float): the minimum height of a peak.
    
    Returns:
        numpy array: a 1D array of peak indices.
    """
    # Extract signal values from the second column
    signal_values = signal[:, 1]
    
    # Find peaks in the signal using the SciPy find_peaks function
    # The height parameter specifies the minimum height of a peak, 
    # and the distance parameter specifies the minimum distance between peaks
    # The negative signal_values are used because find_peaks finds peaks instead of valleys that are looked for in the detrended pressure signal
    peaks, _ = sp.signal.find_peaks(-signal_values, height=threshold, distance = peak_distance)
    
    return peaks

# Define a function to set up plots
def SolPlot(no_peaks, plotflag):
    if plotflag == 1:
        axs = np.empty([no_peaks,9], dtype=object)
        fig = plt.figure(tight_layout=True, figsize=(48, 6 * no_peaks))
        gs = mpl.gridspec.GridSpec(3 * no_peaks, 14)
        for i in range(no_peaks):
            j = i + 1
            
            axs[i,0] = fig.add_subplot(gs[((3*j)-3):(3*j), 0:3])
            axs[i,1] = fig.add_subplot(gs[((3*j)-3):(3*j), 3:6])
            axs[i,2] = fig.add_subplot(gs[((3*j)-3):(3*j), 6:9])

            axs[i,5] = fig.add_subplot(gs[((3*j)-1), 9:11])
            axs[i,3] = fig.add_subplot(gs[((3*j)-3), 9:11],sharex = axs[i,5])
            axs[i,4] = fig.add_subplot(gs[((3*j)-2), 9:11],sharex = axs[i,5])
            plt.setp(axs[i,3].get_xticklabels(), visible=False)
            plt.setp(axs[i,4].get_xticklabels(), visible=False)

            axs[i,8] = fig.add_subplot(gs[((3*j)-1), 11:13])
            axs[i,6] = fig.add_subplot(gs[((3*j)-3), 11:13],sharex = axs[i,8])
            axs[i,7] = fig.add_subplot(gs[((3*j)-2), 11:13],sharex = axs[i,8])
            plt.setp(axs[i,6].get_xticklabels(), visible=False)
            plt.setp(axs[i,7].get_xticklabels(), visible=False)
    else:
        axs = np.array([[1]])
    
    return axs

# Define a function to calculate the lorentzian pressure distribution of a set of points.
def lv_pressure(x, y, P0, x0, y0, R):
    """
    Returns the pressure distribution for a Lorentzian vortex at a given x and y coordinate.

    Parameters:
    x (float or numpy.ndarray): The x coordinate(s) at which to evaluate the pressure.
    y (float or numpy.ndarray): The y coordinate(s) at which to evaluate the pressure.
    P0 (float): The maximum pressure at the center of the vortex.
    x0 (float): The x position of the vortex.
    y0 (float): The y position of the vortex.
    R (float): The FWHM of the vortex.

    Returns:
    P (float or numpy.ndarray): The pressure at the given x and y coordinate(s).
    """
    # calculate the distance from the vortex center to each point (x,y)
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    # calculate the pressure at each point using the Lorentzian vortex pressure formula
    P = -abs(P0) / (1 + ((2 * r) / R)**2)
    return P

# Define a function to calculate the lorentzian velocity distribution of a set of points.
def lv_velocity(x, y, V0, x0, y0, R):
    """
    Returns the velocity distribution for a Lorentzian vortex at a given x and y coordinate.

    Parameters:
    x (float or numpy.ndarray): The x coordinate(s) at which to evaluate the velocity.
    y (float or numpy.ndarray): The y coordinate(s) at which to evaluate the velocity.
    V0 (float): The maximum tangential velocity at the center of the vortex.
    x0 (float): The x position of the vortex.
    y0 (float): The y position of the vortex.
    R (float): The FWHM of the vortex.

    Returns:
    [u v] (float or numpy.ndarray): The u and v velocity at the given x and y coordinate(s).
    """

    # Calculate the distance of each point to the center of the vortex
    r = np.sqrt((x - x0)**2 + (y - y0)**2)

    # Calculate the angle of each point with respect to the center of the vortex
    theta = np.arctan2(y - y0, x - x0)


    # fudge factor to match the wall vortex to the maximum vortex in the profile
    V0 = 2 * V0 / R

    # Calculate the velocity at each point using the Lorentzian vortex equation
    V = (2 * r * V0) / (1 + (((2 * r) / R)**2))

    # Calculate the u and v components of the velocity vector
    u = -V * np.sin(theta)
    v = V * np.cos(theta)
    
    # Return the velocity vector as a numpy array
    return np.array([u, v]).T

# Define function to find time window indexes around a peak in a signal
def define_windows(peaks, window_time, pressure_rate, signal_length):
    """
    Define time windows around each peak in a signal.

    Args:
        peaks (list or numpy array): the indices of the peaks in the signal
        window_time (float): the size of the window to define, in seconds
        pressure_rate (float): the sampling rate of the pressure signal, in Hz
        signal_length (int): the length of the signal, in samples

    Returns:
        A numpy array containing the start and end times (in samples) for each window.
    """
    # Initialize array to hold start and end times for each window
    window_times = np.full([len(peaks), 2], None)


    for i, peak_index in enumerate(peaks):
        # Define the start time for the window
        window_start = np.max([0, peak_index - int((window_time * pressure_rate) / 2)])  

        # Define the end time for the window
        window_end = np.min([signal_length, peak_index + int((window_time * pressure_rate) / 2)])  
        
        # Add the start and end times to the window_times array
        window_times[i] = [window_start, window_end]  
    return window_times

# Define function to find vortex walls in the pressure signal
def define_walls(signal, peaks, window):
    """
    Defines the walls of the vortex ring using the pressure signal data, peaks, and the
    window around the peaks.

    Args:
        signal (numpy array): a 2D array containing the pressure signal data, where the first column
            is time and the second column is pressure.
        peaks (list): a list containing the indices of the peaks in the pressure signal.
        window (numpy array): a 2D array containing the start and end times for the windows around the
            peaks.

    Returns:
        A 2D numpy array containing the start and end indices of the walls for each peak.
    """
    # Extract the pressure values from the signal
    values = signal[:,1]

    # Define the threshold as half the peak value
    threshold = values[peaks] / 2.0

    # Initialize an array to store the wall indices
    walls = np.full([len(peaks),2],None)

    # Loop through each peak and find the walls
    for ii, peak in enumerate(peaks):
        # Search for the first time the signal crosses the threshold before the index
        for i in range(peak, window[ii,0], -1):
            if values[i] <= threshold[ii] and values[i-1] >= threshold[ii]:
                walls[ii,0] = i
                break

        # Search for the first time the signal crosses the threshold after the index
        for i in range(peak, window[ii,1]):
            if values[i] <= threshold[ii] and values[i+1] >= threshold[ii]:
                walls[ii,1] = i
                break

    return walls

# Define function to apply lorentzian vortex to peak
def lor_pres_at_peak(signal, wall, window, peak):
    """
    Compute the Lorentzian pressure at the peak of the signal within the specified window.

    Args:
        signal (numpy array): an array containing the time and pressure data.
        wall (numpy array): an array containing the start and end indices of the wall region.
        window (numpy array): an array containing the start and end indices of the window region.
        peak (int): the index of the peak in the signal.

    Returns:
        A tuple containing the time radius of the Lorentzian and the pressure values of the Lorentzian
        at the specified peak.
    """
    # Compute the time radius of the signal
    signal_time_radius = signal[wall[1], 0] - signal[wall[0], 0]

    # Compute the Lorentzian pressure within the specified window and at the peak of the signal
    signal_out = lv_pressure(signal[window[0]:window[1], 0], 0, signal[peak, 1], signal[peak, 0], 0, signal_time_radius)

    return signal_time_radius, signal_out

# Define function to apply velocity function to peak
def lor_vel_at_peak(signal,wall,window,peak_time):
    """
    Calculate the velocity of a Lorentzian vortex at the peak.

    Args:
        signal (numpy array): an array containing the signal data.
        wall (numpy array): an array containing the indices of the start and end of the wall.
        window (numpy array): an array containing the indices of the start and end of the window.
        peak_time (float): the time at which the peak occurs.

    Returns:
        A tuple containing the velocity at the peak, the time radius of the signal, and an array of velocities over time
        calculated using the Lorentzian vortex velocity function.
    """
    # Compute the signal time radius
    signal_time_radius = signal[wall[1],0] - signal[wall[0],0]

    # Compute the velocity peak by taking the average of the absolute values of the velocity at the two walls
    velpeak = np.abs(signal[wall[0],1]) + np.abs(signal[wall[1],1]) / 2

    # Compute the Lorentzian vortex velocity at the peak time using the signal time radius, velocity peak, and peak time
    signal_out = lv_velocity(signal[window[0]:window[1],0],0,velpeak,peak_time,0,signal_time_radius)

    return velpeak, signal_time_radius, signal_out

# Define fucntion to match index of two signals at given time
def time_match_index(signal1, signal2, window, tolerance=1e-3):
    """
    Match the indices of two signals at given time windows.

    Args:
        signal1 (numpy array): an array containing the first signal's time and amplitude data.
        signal2 (numpy array): an array containing the second signal's time and amplitude data.
        window (numpy array): an array containing the start and end indices of each time window.
        tolerance (float): the tolerance for matching time indices in decimal sol

    Returns:
        A numpy array containing the matched indices of the second signal.
    """
    match_index = np.full_like(window,None)
    for i, v in enumerate(window[:,0]):
        # Find the index of the closest value in signal2 to the start and end of the window in signal1
        idx1 = np.abs(signal2[:,0] - signal1[window[i,0],0]).argmin()
        match_index[i,0] = idx1
        idx2 = np.abs(signal2[:,0] - signal1[window[i,1],0]).argmin()
        match_index[i,1] = idx2
        if np.abs(signal2[idx1,0] - signal1[window[i,0],0]) > tolerance:
            match_index[i,0] = 0
        if np.abs(signal2[idx2,0] - signal1[window[i,1],0]) > tolerance:
            match_index[i,1] = 0
    return match_index

# Define velocity residual function
def vel_residuals(p, ydata, xdata, R):
    """
    Compute the residuals between the observed data and the model predictions using
    the Lorentzian vortex velocity function.
    
    Args:
        p (tuple): a tuple containing the initial guess for the parameters of the Lorentzian
            vortex velocity function, specifically the position x0 and the maximum velocity V0.
        ydata (numpy array): an array containing the observed y-values (i.e. velocity) of the data.
        xdata (numpy array): an array containing the x-values of the data.
        R (float): the FWHM of the vortex ring.
    
    Returns:
        An array of residuals (i.e. the differences between the observed y-values and the model predictions).
    """
    
    # Unpack parameters
    x0, V0 = p
    
    # Compute the model predictions using the Lorentzian vortex velocity function, using only the v component of velocity
    ymodel = lv_velocity(xdata, 0, V0, x0, 0, R)[:,1]
    
    # Compute the residuals (i.e. the differences between the observed y-values and the model predictions)
    res = ymodel - ydata
    
    # Remove any NaN values from the residuals array
    res = res[~np.isnan(res)]
    
    # Return the array of residuals
    return res

# Define pressure residual function
def pres_residuals(p, ydata, xdata, R):
    """
    Compute the residuals between the observed data and the model predictions using
    the Lorentzian vortex pressure function.
    
    Args:
        p (tuple): a tuple containing the initial guess for the parameters of the Lorentzian
            vortex pressure function, specifically the position x0 and the maximum pressure P0.
        ydata (numpy array): an array containing the observed y-values (i.e. velocity) of the data.
        xdata (numpy array): an array containing the x-values of the data.
        R (float): the FWHM of the vortex ring.
    
    Returns:
        An array of residuals (i.e. the differences between the observed y-values and the model predictions).
    """
    
    # Unpack parameters
    x0, P0 = p
    
    # Compute the model predictions using the Lorentzian vortex pressure function
    ymodel = lv_pressure(xdata, 0, P0, x0, 0, R)
    
    # Compute the residuals (i.e. the differences between the observed y-values and the model predictions)
    res = ymodel - ydata
    
    # Remove any NaN values from the residuals array
    res = res[~np.isnan(res)]
    
    # Return the array of residuals
    return res

# Define pressure residual2 function
def pres_residuals2(p, ydata, xdata):
    """
    Compute the residuals between the observed data and the model predictions using
    the Lorentzian vortex pressure function.
    
    Args:
        p (tuple): a tuple containing the initial guess for the parameters of the Lorentzian
            vortex pressure function, specifically the position x0 and the maximum pressure P0.
        ydata (numpy array): an array containing the observed y-values (i.e. velocity) of the data.
        xdata (numpy array): an array containing the x-values of the data.
    
    Returns:
        An array of residuals (i.e. the differences between the observed y-values and the model predictions).
    """
    
    # Unpack parameters
    x0, P0, R = p
    
    # Compute the model predictions using the Lorentzian vortex pressure function
    ymodel = lv_pressure(xdata, 0, P0, x0, 0, R)
    
    # Compute the residuals (i.e. the differences between the observed y-values and the model predictions)
    res = ymodel - ydata
    
    # Remove any NaN values from the residuals array
    res = res[~np.isnan(res)]
    
    # Return the array of residuals
    return res

# Define a function to determine how well the lorentzian fits the data    
def goodness_of_fit(signal1, signal2):
    """
    Computes the mean squared error between two signals.

    Args:
        signal1 (numpy array): the first signal.
        signal2 (numpy array): the second signal.

    Returns:
        mse (float): the mean squared error between the two signals.
    """
    # Compute the mean squared error between the two signals
    mse = np.nanmean((signal1 - signal2)**2)
    return mse

#Define a function to determine the centre characteristice of the vortex
def pass_distance(i):
    theta_obs = key_data.loc[i,'theta_obs']
    delta_P_obs = np.abs(key_data.loc[i,'delta_P_obs'])
    delta_P_obs = np.abs(key_data.loc[i,'p_lsr_delta_P'])
    tau_obs = key_data.loc[i,'time_FWHM'] * (87775.2440/86400)  # convert LTST seconds to actual seconds
    V = key_data.loc[i,'backg_V']
    E = 2.7e8
    nu = 0.22
    eta = E / (1 - nu**2)

    closest_approach = np.sqrt((delta_P_obs * (tau_obs * V)**2) / (eta * theta_obs))

    D = np.sqrt((tau_obs * V)**2 * (1-(4*delta_P_obs)/(eta *theta_obs)))

    delta_P = -1 * (eta * theta_obs * delta_P_obs) / ((eta * theta_obs) - (4 * delta_P_obs))
    
    key_data.loc[i,'miss_distance'] = closest_approach
    key_data.loc[i,'D'] = D
    key_data.loc[i,'delta_P'] = delta_P

def pressure_plot(axs, pressure_dt, pressure_bckgnd, window_times, wall_locs, pressure_rate, window_add, peak, i, plotflag):
    global key_data
    if plotflag == 1:
        axs[i,0].plot(pressure_dt[window_times[i,0]:window_times[i,1],0],pressure_dt[window_times[i,0]:window_times[i,1],1], color = '#1f77b4')
    pres_strad, pres_vortex_fit = lor_pres_at_peak(pressure_dt,wall_locs[i,:],window_times[i,:],peak)
    if plotflag == 1:
        #axs[i,0].plot(pressure_dt[window_times[i,0]:window_times[i,1],0],pres_vortex_fit,color = 'blue')
        axs[i,0].axvline(pressure_dt[peak,0],color = 'black',linestyle = '--')
        #yyaxs = axs[i,0].twinx()
        #yyaxs.plot(pressure_dt[window_times[i,0]:window_times[i,1],0], pressure_bckgnd[window_times[i,0]:window_times[i,1]],color = '#33fff5')
    #p_guess = [pressure_dt[peak,0], pressure_dt[peak,1]]
    #p_fit = sp.optimize.least_squares(pres_residuals, p_guess, args=(pressure_dt[window_times[i,0]:window_times[i,1],1], pressure_dt[window_times[i,0]:window_times[i,1],0],pres_strad))
    #x0_fit, P0_fit = p_fit.x
    #pres_lsr = lv_pressure(pressure_dt[window_times[i,0]:window_times[i,1],0], 0, P0_fit,x0_fit,0,pres_strad)

    p_guess2 = [pressure_dt[peak,0], pressure_dt[peak,1], pres_strad]
    p_fit = sp.optimize.least_squares(pres_residuals2, p_guess2, args=(pressure_dt[window_times[i,0]:window_times[i,1],1], pressure_dt[window_times[i,0]:window_times[i,1],0]))
    x0_fit, P0_fit, R0_fit = p_fit.x
    pres_lsr = lv_pressure(pressure_dt[window_times[i,0]:window_times[i,1],0], 0, P0_fit,x0_fit,0, R0_fit)

    if plotflag ==1:
        axs[i,0].plot(pressure_dt[window_times[i,0]:window_times[i,1],0], pres_lsr)
        
        axs[i,0].set_ylabel(r'$\Delta$P [Pa]')
        axs[i,0].set_xlabel(r'Time')

    key_data.loc[i,'delta_P_obs'] = pressure_dt[peak, 1]
    #key_data.loc[i,'p_lor_fit'] = goodness_of_fit(pressure_dt[window_times[i,0]:window_times[i,1],1],pres_vortex_fit)
    key_data.loc[i,'p_lsr_delta_P'] = P0_fit
    key_data.loc[i,'p_lsr_peak_time'] = x0_fit
    key_data.loc[i,'p_lsr_fit'] = goodness_of_fit(pressure_dt[window_times[i,0]:window_times[i,1],1],pres_lsr)
    key_data.loc[i,'pres_rate'] = pressure_rate
    #key_data.loc[i,'time_FWHM'] = pres_strad * 86400.0
    key_data.loc[i,'time_FWHM'] = R0_fit * 86400.0

    window_data = pressure_dt[window_times[i,0]-int(pressure_rate*window_add):window_times[i,1]+int(pressure_rate*window_add),[0,1]]

    pres_out.append(window_data)

def velocity_plot(axs, pressure_dt, twins_dt, twins_bckgnd, vel_window_times, vel_wall_locs, peak, i, plotflag):
    global key_data
    if plotflag == 1:
        axs[i,1].plot(twins_dt[vel_window_times[i,0]:vel_window_times[i,1],0],twins_dt[vel_window_times[i,0]:vel_window_times[i,1],1], color = '#b41f76')
    # V0_init, vel_strad, vel_vortex_fit = lor_vel_at_peak(twins_dt,vel_wall_locs[i,:],vel_window_times[i,:],pressure_dt[peak,0])
    if plotflag == 1:
        #axs[i,1].plot(twins_dt[vel_window_times[i,0]:vel_window_times[i,1],0],vel_vortex_fit[:,1],color = 'red')
        axs[i,1].axvline(pressure_dt[peak,0],color = 'black',linestyle = '--')
        #yyaxs = axs[i,1].twinx()
        #yyaxs.plot(twins_dt[vel_window_times[i,0]:vel_window_times[i,1],0], twins_bckgnd[vel_window_times[i,0]:vel_window_times[i,1],0],color = '#ffae33')
    #if np.isnan(V0_init):
    #    V0_init = 1
    # p_guess = [pressure_dt[peak,0], V0_init]
    # p_fit = sp.optimize.least_squares(vel_residuals, p_guess, args=(twins_dt[vel_window_times[i,0]:vel_window_times[i,1],1], twins_dt[vel_window_times[i,0]:vel_window_times[i,1],0],vel_strad))
    # x0_fit, V0_fit = p_fit.x
    # vel_lsr = lv_velocity(twins_dt[vel_window_times[i,0]:vel_window_times[i,1],0], 0, V0_fit,x0_fit,0,vel_strad)[:,1]
    if plotflag == 1:
        #axs[i,1].plot(twins_dt[vel_window_times[i,0]:vel_window_times[i,1],0], vel_lsr)

        axs[i,1].set_ylabel(r'V [m/s]')
        axs[i,1].set_xlabel(r'Time')

    key_data.loc[i,'backg_V'] = np.nanmean(twins_bckgnd[vel_window_times[i,0]:vel_window_times[i,1],1])
    # key_data.loc[i,'delta_V'] = V0_init
    # key_data.loc[i,'v_lor_fit'] = goodness_of_fit(twins_dt[vel_window_times[i,0]:vel_window_times[i,1],1],vel_vortex_fit[:,1])
    # key_data.loc[i,'v_lsr_delta_V'] = V0_fit
    # key_data.loc[i,'v_lsr_peak_time'] = x0_fit
    # key_data.loc[i,'v_lsr_fit'] = goodness_of_fit(twins_dt[vel_window_times[i,0]:vel_window_times[i,1],1],vel_lsr)

def temp_eng_plot(axs, pressure_dt, temperature, eng_data, temp_window_times, cur_window_times, peak, i, plotflag):
    if plotflag == 1:
        axs[i,2].plot(temperature[temp_window_times[i,0]:temp_window_times[i,1],0],temperature[temp_window_times[i,0]:temp_window_times[i,1],1])
        axs[i,2].axvline(pressure_dt[peak,0],color = 'black',linestyle = '--')
        yyaxis = axs[i,2].twinx()
        yyaxis.plot(eng_data[cur_window_times[i,0]:cur_window_times[i,1],0],eng_data[cur_window_times[i,0]:cur_window_times[i,1],4],color = 'purple')
        axs[i,2].set_ylabel(r'T [K]')
        axs[i,2].set_xlabel(r'Time')  

def magnetic_plot(axs, pressure_dt, magnetic_dt, magnetic_bckgnd, magnetic_window_times, magnetic_rate, window_add, peak, i, plotflag,med_flag):
    global key_data
    global mag_out
    global mag_bckgnd
    if plotflag == 1:
        axs[i,3].plot(magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],0],magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],1], color = '#A2142F')
        axs[i,3].axvline(pressure_dt[peak,0],color = 'black',linestyle = '--')
        axs[i,3].set_ylabel('N')
        axs[i,3].set_ylabel(r'MFD [nT]')
        yyaxis = axs[i,3].twinx()
        yyaxis.plot(magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],0],magnetic_bckgnd[magnetic_window_times[i,0]:magnetic_window_times[i,1],0])

        axs[i,4].plot(magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],0],magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],2], color = "#77AC30")
        axs[i,4].axvline(pressure_dt[peak,0],color = 'black',linestyle = '--')
        axs[i,4].set_ylabel(r'MFD [nT]')
        yyaxis = axs[i,4].twinx()
        yyaxis.plot(magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],0],magnetic_bckgnd[magnetic_window_times[i,0]:magnetic_window_times[i,1],1])
        
        axs[i,5].plot(magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],0],magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],3], color = "#7E2F8E")
        axs[i,5].axvline(pressure_dt[peak,0],color = 'black',linestyle = '--')
        axs[i,5].set_ylabel(r'MFD [nT]')
        axs[i,5].set_xlabel(r'Time')
        yyaxis = axs[i,5].twinx()
        yyaxis.plot(magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],0],magnetic_bckgnd[magnetic_window_times[i,0]:magnetic_window_times[i,1],1])
    
    if med_flag ==1:
        med_set = magnetic_dt[magnetic_window_times[i,0]-int(200*magnetic_rate):magnetic_window_times[i,0]-1,3]
        mag_median = np.median(med_set,axis=0)
        magnetic_dt[:,1:] = magnetic_dt[:,1:] - mag_median

    key_data.loc[i,'peak_Bz'] = max(abs(magnetic_dt[magnetic_window_times[i,0]:magnetic_window_times[i,1],3]))
    key_data.loc[i,'mag_rate'] = magnetic_rate

    col_name = f'SOL{key_data.Sol[i]}_Peak{i}'
    window_data = magnetic_dt[magnetic_window_times[i,0]-int(magnetic_rate*window_add):magnetic_window_times[i,1]+int(magnetic_rate*window_add),:]
    #background_seconds = 250
    background_seconds = 1000
    background_data = magnetic_dt[(magnetic_window_times[i,0]-int(magnetic_rate*background_seconds)):magnetic_window_times[i,0],:] 
    mag_out.append(window_data)
    mag_bckgnd.append(background_data)

def seismic_plot(axs, pressure_dt, seis_dt, seis_bckgnd, seis_window_times, peak, i, plotflag):
    times = seis_dt[seis_window_times[i,0]:seis_window_times[i,1],0]
    a_N = seis_dt[seis_window_times[i,0]:seis_window_times[i,1],1]
    a_E = seis_dt[seis_window_times[i,0]:seis_window_times[i,1],2]
    a_Z = seis_dt[seis_window_times[i,0]:seis_window_times[i,1],3]
    if plotflag == 1:
        axs[i,6].plot(times, a_N, color = '#A2142F')
        axs[i,6].axvline(pressure_dt[peak,0],color = 'black',linestyle = '--')
        axs[i,6].set_ylabel(r'a [m/s$^2$]')
        #yyaxis = axs[i,6].twinx()
        #yyaxis.plot(times,seis_bckgnd[seis_window_times[i,0]:seis_window_times[i,1],0])

        axs[i,7].plot(times, a_E, color = "#77AC30")
        axs[i,7].axvline(pressure_dt[peak,0],color = 'black',linestyle = '--')
        axs[i,7].set_ylabel(r'a [m/s$^2$]')
        #yyaxis = axs[i,7].twinx()
        #yyaxis.plot(times,seis_bckgnd[seis_window_times[i,0]:seis_window_times[i,1],1])

        axs[i,8].plot(times, a_Z, color = "#7E2F8E")
        axs[i,8].axvline(pressure_dt[peak,0],color = 'black',linestyle = '--')
        axs[i,8].set_ylabel(r'a [m/s$^2$]')
        axs[i,8].set_xlabel(r'Time')
        #yyaxis = axs[i,8].twinx()
        #yyaxis.plot(times,seis_bckgnd[seis_window_times[i,0]:seis_window_times[i,1],2])

    g = 3.71

    theta_obs = np.max(np.sqrt(np.power(a_E, 2) + np.power(a_N, 2)) / g)

    max_index = np.argmax(np.sqrt(np.power(a_E, 2) + np.power(a_N, 2)))

    azimuth = np.arctan2(a_E[max_index],a_N[max_index])

    key_data.loc[i,'theta_obs'] = theta_obs
    key_data.loc[i,'azimuth'] = azimuth

# Define a function to find all the window indexes and wall locations for all signals to use 
def find_windows_walls(pressure_dt, twins_dt, magnetic_dt, temperature, eng_data, seis_dt, window_time, pressure_rate, peaks, pres_len, magnetic_distance):
    window_times = define_windows(peaks, window_time, pressure_rate, pres_len)
    
    ext_window_times = define_windows(peaks, magnetic_distance, pressure_rate, pres_len)
    wall_locs = define_walls(pressure_dt, peaks, window_times)

    peaks = np.delete(peaks, np.where(np.any(wall_locs == None, axis=1))[0], axis = 0)
    
    window_times = np.delete(window_times, np.where(np.any(wall_locs == None, axis=1))[0], axis = 0)
    ext_window_times = np.delete(ext_window_times, np.where(np.any(wall_locs == None, axis=1))[0], axis = 0)

    wall_locs = np.delete(wall_locs, np.where(np.any(wall_locs == None, axis=1))[0], axis = 0)

    vel_window_times = time_match_index(pressure_dt,twins_dt,window_times)

    vel_wall_locs = time_match_index(pressure_dt,twins_dt,wall_locs)

    cur_window_times = time_match_index(pressure_dt, eng_data, window_times)

    temp_window_times = time_match_index(pressure_dt, temperature, window_times)

    #magnetic_window_times = time_match_index(pressure_dt, magnetic_dt, window_times)
    magnetic_window_times = time_match_index(pressure_dt, magnetic_dt, ext_window_times)

    seis_window_times = time_match_index(pressure_dt, seis_dt, window_times)
    #seis_window_times = time_match_index(pressure_dt, seis_dt, wall_locs)

    mask = (
        (window_times != 0).all(axis=1) &
        (wall_locs != 0).all(axis=1) &
        (vel_window_times != 0).all(axis=1) &
        (vel_wall_locs != 0).all(axis=1) &
        #(cur_window_times != 0).all(axis=1) &
        (temp_window_times != 0).all(axis=1) &
        (magnetic_window_times != 0).all(axis=1) &
        (seis_window_times != 0).all(axis=1)
    )

    # Apply mask to all arrays
    peaks = peaks[mask]
    window_times = window_times[mask]
    wall_locs = wall_locs[mask]
    vel_window_times = vel_window_times[mask]
    vel_wall_locs = vel_wall_locs[mask]
    cur_window_times = cur_window_times[mask]
    temp_window_times = temp_window_times[mask]
    magnetic_window_times = magnetic_window_times[mask]
    seis_window_times = seis_window_times[mask]

    return window_times, peaks, wall_locs, vel_window_times, vel_wall_locs, temp_window_times, cur_window_times, magnetic_window_times, seis_window_times

# Define the main function - this is where everything happens
def Sol(SolNumber, plotflag):
    global key_data
    global mag_out
    global pres_out
    global mag_bckgnd
    key_data = pd.DataFrame(columns =['Sol','peak_centre',
                                'delta_P_obs','p_lor_fit','p_lsr_delta_P','p_lsr_peak_time','p_lsr_fit', 'pres_rate',
                                'delta_V','v_lor_fit','v_lsr_delta_V','v_lsr_peak_time','v_lsr_fit','backg_V',
                                'time_FWHM','theta_obs', 'azimuth',
                                'miss_distance','D','delta_P',
                                'peak_Bz', 'mag_rate'])
    mag_out = []
    pres_out = []
    mag_bckgnd = []
    try:
        average_distance = 1000

        window_time = 200
        magnetic_distance = 1000
        window_add = 0.1 * window_time

        print('Sol', SolNumber)
        logging.info(f"Sol {SolNumber}")
        paths = SolPath(SolNumber)
        if paths:
            magnetic, magnetic_rate, pressure, pressure_rate, temperature, twins, twins_rate, seis, seis_rate, eng_data = data_extract(paths)
            
            pressure, pressure_rate = lowpass_2Hz(pressure,pressure_rate)
            seis, seis_rate = downsample(seis, seis_rate, 5)
            #twins, twins_rate = downsample(twins, twins_rate, 2)

            seis = bandpass(seis, seis_rate, 0.02, 0.3)

            # added for naomi
            #pressure = bandpass(pressure, pressure_rate, 0.02, 0.3)

            seis = differentiate(seis, seis_rate)

            #magnetic_dt, magnetic_bckgnd = detrend(magnetic, magnetic_rate, magnetic_distance)
            magnetic_dt = magnetic
            magnetic_bckgnd = magnetic[:,1:]
            med_flag = 1
            pressure_dt, pressure_bckgnd = detrend(pressure, pressure_rate, average_distance)
            #twins_dt, twins_bckgnd = detrend(twins, twins_rate, 200)

            twins_dt = time_average(twins,twins_rate, 200)
            twins_bckgnd = twins_dt
            
            #seis_dt, seis_bckgnd = detrend(seis, seis_rate, average_distance)
            seis_dt = seis
            seis_bckgnd = seis

            peak_distance = 50 * pressure_rate
            peak_threshold = 0.35
            peaks = find_signal_peaks(pressure_dt, peak_threshold, peak_distance)

            pres_len = len(pressure[:,1])
            window_times, peaks, wall_locs, vel_window_times, vel_wall_locs, temp_window_times, eng_window_times, magnetic_window_times, seis_window_times = find_windows_walls(pressure_dt, twins_dt, magnetic_dt, temperature, eng_data, seis_dt, window_time, pressure_rate, peaks, pres_len, magnetic_distance)
            
            print(f'Sol {SolNumber}:', len(peaks), 'peaks found')
            logging.info(f"Sol {SolNumber}: {len(peaks)} peaks found")

            axs = SolPlot(len(peaks),plotflag)

            for i, peak in enumerate(peaks):

                key_data.loc[i,'Sol'] = SolNumber
                key_data.loc[i,'peak_centre'] = pressure_dt[peak,0]

                pressure_plot(axs,pressure_dt, pressure_bckgnd, window_times, wall_locs, pressure_rate, window_add,peak, i, plotflag)
                
                velocity_plot(axs, pressure_dt, twins_dt, twins_bckgnd, vel_window_times, vel_wall_locs, peak, i, plotflag)
                
                temp_eng_plot(axs, pressure_dt, temperature, eng_data, temp_window_times, eng_window_times, peak, i, plotflag)

                magnetic_plot(axs, pressure_dt, magnetic_dt, magnetic_bckgnd, magnetic_window_times, magnetic_rate, window_add, peak, i, plotflag, med_flag)

                seismic_plot(axs, pressure_dt, seis_dt, seis_bckgnd, seis_window_times, peak, i, plotflag)

                pass_distance(i)

            max_length = max(data.shape[0] for data in mag_out)
            mag_out = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant') for data in mag_out]
            mag_out = np.concatenate(mag_out, axis=1)
            mag_out = pd.DataFrame(mag_out)

            max_length = max(data.shape[0] for data in pres_out)
            pres_out = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant') for data in pres_out]
            pres_out = np.concatenate(pres_out, axis=1)
            pres_out = pd.DataFrame(pres_out)

            max_length = max(data.shape[0] for data in mag_bckgnd)
            mag_bckgnd = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant') for data in mag_bckgnd]
            mag_bckgnd = np.concatenate(mag_bckgnd, axis=1)
            mag_bckgnd = pd.DataFrame(mag_bckgnd)

            column_names = [f'Sol{key_data.Sol[i]}_Peak{i}_{suffix}' for i, peak in enumerate(peaks) for suffix in ['time', 'data']]
            column_names_mag = [f'Sol{key_data.Sol[i]}_Peak{i}_{suffix}' for i, peak in enumerate(peaks) for suffix in ['time', 'dataN','dataE','dataZ']]
            mag_out.columns = column_names_mag
            pres_out.columns = column_names
            mag_bckgnd.columns = column_names_mag

            print('Sol', SolNumber, 'Complete')
            logging.info(f"Sol {SolNumber} Complete")
            if plotflag == 1:
                plt.show()
            return key_data, mag_out, pres_out, mag_bckgnd

        else:
            print('Sol', SolNumber, ': Insufficient Data')
            mag_out = pd.DataFrame(mag_out).T
            pres_out = pd.DataFrame(pres_out).T
            mag_bckgnd = pd.DataFrame(pres_out).T
            logging.info(f"Sol {SolNumber}: Insufficient Data")
            return key_data, mag_out, pres_out,mag_bckgnd
    except Exception as e:
        key_data = []
        mag_out = []
        pres_out = []
        mag_bckgnd = []
        key_data = pd.DataFrame(key_data).T
        mag_out = pd.DataFrame(mag_out).T
        pres_out = pd.DataFrame(pres_out).T
        mag_bckgnd = pd.DataFrame(mag_bckgnd).T
        print(f"Error occurred in Sol {SolNumber}: {e}")
        logging.error(f"Error occurred in Sol {SolNumber}: {e}")
        return key_data, mag_out, pres_out,mag_bckgnd

# Define function to return 186 non peak events for signal windowing
def Sol_mag_window(SolNumber, num_events):
    mag_out = []
    try:
        average_distance = 500
        peak_distance = 500
        peak_threshold = 0.35
        window_time = 200
        window_add = 0.1 * window_time

        print('Sol', SolNumber)
        logging.info(f"Sol {SolNumber}")
        paths = SolPath(SolNumber)
        if paths:
            magnetic, magnetic_rate, pressure, pressure_rate, temperature, twins, twins_rate, seis, seis_rate, eng_data = data_extract(paths)
            magnetic_dt = magnetic
            pressure_dt, _ = detrend(pressure, pressure_rate, average_distance)
            twins_dt = time_average(twins,twins_rate, 200)
            
            #seis_dt, seis_bckgnd = detrend(seis, seis_rate, average_distance)
            seis_dt = seis

            peaks = find_signal_peaks(pressure_dt, peak_threshold, peak_distance)
            
            pres_len = len(pressure[:,1])
            _, peaks, _, _, _, _, _, magnetic_window_times, _ = find_windows_walls(pressure_dt, twins_dt, magnetic_dt, temperature, eng_data, seis_dt, window_time, pressure_rate, peaks, pres_len)
            
            magnetic_window_times[:,0] = magnetic_window_times[:,0] - int(magnetic_rate*window_add) 
            magnetic_window_times[:,1] = magnetic_window_times[:,1] + int(magnetic_rate*window_add)

            events = np.zeros(shape=(num_events),dtype=np.int64)
            for i in range(0,num_events):
                loop = True
                centre_lim = np.zeros(shape=(2),dtype=np.int64)
                while loop:
                    centre = np.random.uniform(0.35,0.65) + SolNumber
                    centre = int(np.abs(centre - magnetic_dt[:,0]).argmin())
                    centre_lim[0] = centre - int(window_time * magnetic_rate)
                    centre_lim[1] = centre + int(window_time * magnetic_rate)
                    success_count = 0
                    
                    for j, _ in enumerate(peaks):
                        if (centre_lim[0] > magnetic_window_times[j,0] and centre_lim[1] > magnetic_window_times[j,1]) or ((centre_lim[0] < magnetic_window_times[j,0] and centre_lim[1] < magnetic_window_times[j,1])):
                            success_count += 1
                    if success_count == j+1:
                        loop = False
                        events[i] = centre
            
            for i in range(0,num_events):
                window_data = magnetic_dt[events[i] - int(window_time * magnetic_rate):events[i] + int(window_time * magnetic_rate),[0,3]]
                mag_out.append(window_data)

        max_length = max(data.shape[0] for data in mag_out)
        mag_out = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant') for data in mag_out]
        mag_out = np.concatenate(mag_out, axis=1)
        mag_out = pd.DataFrame(mag_out)
        
        column_names = [f'Sol{SolNumber}_Event{i}_{suffix}' for i in range(0,num_events) for suffix in ['time', 'data']]
        mag_out.columns = column_names
        return mag_out

    except Exception as e:
        mag_out = []
        mag_out = pd.DataFrame(mag_out).T
        print(f"Error occurred in Sol {SolNumber}: {e}")
        logging.error(f"Error occurred in Sol {SolNumber}: {e}")
        return mag_out