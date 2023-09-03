import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import shift

def t_nmo(offset, t0=0.2, v_nmo=4000):
    '''
    Calculate NMO Time
    
    Input:
    offset = Distance location of each trace [m]. Type: 1-D list or array.
    t0 = Time at offset=0 or starting time of the trace [s]. Type: float. Default: 0.2.
    v_nmo = NMO velocity that shaped the curve [m/s]. Type: float. Default: 4000.
    
    Output:
    Arrival time of the trace [s]. Type: 1-D array with same size as offset.
    '''
    return np.sqrt(t0**2 + (offset**2)/(v_nmo**2))
	
def RickerWavelet(f=25, l=0.128, dt=0.002):
    '''
    Create Ricker (Mexican hat) wavelet to be convoluted with the reflectivity.
    
    Input:
    f = Wavelet frequency [Hz]. Lower is wider. Type: float. Default: 25.
    l = Time length [s]. Type: float. Default: 0.128.
    dt = Sampling rate [s]. Type: float. Default: 0.002 (2 ms).
    
    Output:
    t = Wavelet time [s]. Type: 1-D array.
    y = Wavelet amplitude. Maximum 1. Minimum < 0. Type: 1-D array.
    '''
    t = np.arange(-l/2, (l-dt)/2, dt)
    y = (1 - 2*((np.pi)**2)*(f**2)*(t**2)) / np.exp(((np.pi)**2)*(f**2)*(t**2))
    return t, y

def create_synthetic(time, velocity, channel, offset, dt=0.002, wavelet_freq=25):
    '''
    Create synthetic seismogram from NMO time. Convert the time to the reflectivity amplitude,
    then, convoluted with the wavelet.
    
    Input:
    time = Zero-offset time of the trace [s]. Type: 1-D list or array.
    velocity = NMO velocity of the curve [m/s]. Type: 1-D list or array.
    ***Length of time and velocity are MUST the same. It represents the layers.***
    channel = Number of channel (trace). Type: int.
    dt = Sampling rate [s]. Type: float. Default: 0.002 (2 ms).
    wavelet_freq = Wavelet frequency [Hz]. Lower is wider. Type: float. Default: 25.
    
    Output:
    amplitude = Amplitude matrix of each traces over time. Type: 2-D array with size (time/dt) x channel.
    Y = Time coordinate of one trace [s]. Type: 1-D array with size time/dt.
    '''
    t = np.zeros((len(time), channel))
    vel = np.zeros((len(velocity), channel))
    
    for i in range(len(vel)):
        vel[i].fill(velocity[i])
        t[i].fill(time[i])
        
    NMO = t_nmo(offset, t0=t, v_nmo=vel)
    
    X = np.arange(0,channel,1)
    Y = np.arange(0, np.max(NMO).round(), dt)
    time_matrix = np.meshgrid(X,Y)[1]
    
    target = np.array([np.abs(Y-i).argmin() for i in NMO.flatten()]).reshape(NMO.shape)
    
    amplitude_matrix = np.zeros(time_matrix.shape)
    
    for i in range(NMO.shape[1]):
        time_matrix[target[:,i],i] = NMO[:,i]
        amplitude_matrix[target[:,i],i] = 1
        
    t1, y1 = RickerWavelet(f=wavelet_freq)
    
    amplitude = np.array([np.convolve(amplitude_matrix[:,i],y1,'same') for i in range(amplitude_matrix.shape[1])]).T
    
    return amplitude, Y

def plot_seismogram(ax, data, offset, time, plot_option=1, cmap='seismic'):
    '''
    Simplified function to easily plot the seismogram.
    
    Input:
    ax = matplot.pyplot.Axes of the plot. Might be declared before or not, depends on the plot_option. 
    data = Seismogram amplitude matrix data over the time. Type: 2-D array.
    offset = Distance location of each trace [m]. Type: 1-D list or array with size of data.shape[1].
    time = Time coordinate of one trace [s]. Type: 1-D array with size of data.shape[0].
    plot_option = Plot view option of the seismogram. Explained in the function comment below. Type: int or str. Default: 1.
    cmap = Colormap of the seismogram. Not necessary if plot_option = 0 or 3 or 'wiggle' or 'stacked'. Type: str.
    ***Further options of cmap : https://matplotlib.org/stable/tutorials/colors/colormaps.html***
    
    If the plot_option is not 2 or 'both', the ax must be declared before.
    Example:
    fig, ax = plt.subplots()
    plot_seismogram(ax, your_data, your_offset, your_time, plot_option=1, cmap='seismic')
    
    Output:
    matplotlib.pyplot show of the seismogram plot (no return).
    '''
    ### Wiggle trace
    def wiggle(ax, data, time, plot_option=plot_option):
        for i in range(amplitude.shape[1]):
            ax.plot(amplitude[:,i]+i, Y, linewidth=0.2, c='k')
            ax.fill_between(amplitude[:,i]+i,Y,0, where=(amplitude[:,i]+i>=i) ,lw=0, color='k')
        
        if plot_option == 0 or plot_option == 'wiggle':
            ax.invert_yaxis()
        
        ax.set_ylabel('Time [s]')
        ax.set_xlabel('Trace number')
        
    ### Filled color trace
    def color(ax, data, time, offset, plot_option=plot_option):
        ax.imshow(data, cmap=cmap,
                  extent=[np.min(offset), np.max(offset),
                          np.max(time), np.min(time)],
                  aspect='auto', vmin=-100, vmax=100)
        
        if plot_option == 1 or plot_option == 'color':
            ax.set_ylabel('Time [s]')
            
        ax.set_xlabel('Offset [m]')
    
    ### Stacked (single) trace of the data input
    def stack_seismogram(ax, data, time, offset, plot_option=plot_option):
        stacked = data.sum(axis=1)
        
        positive_amp = stacked.copy()
        positive_amp[positive_amp<0] = 0
        
        ax.plot(stacked, time, lw=0.5, color='k')
        ax.fill_between(positive_amp, time, color='k', lw=0)
        ax.set_xlabel('Amplitude')
    
    # Plot the wiggle trace only. ax must be declared before
    if plot_option == 0 or plot_option == 'wiggle':
        wiggle(ax, data, time)
    # Plot the filled color trace only. ax must be declared before
    elif plot_option == 1 or plot_option == 'color':
        color(ax, data, time, offset)
    # Plot both of wiggle and filled color trace. ax does not need to be declared before.
    elif plot_option == 2 or plot_option == 'both':
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        wiggle(ax[0], data, time)
        color(ax[1], data, time, offset)
    # Plot the stacked trace only. ax must be declared before
    elif plot_option == 3 or plot_option == 'stacked':
        stack_seismogram(ax, data, time, offset)

def shift_nmo(data, t0, v, time, offset):
    '''
    Shifting the seismogram matrix based on the velocity and time from NMO.
    This code was modified from "Geophysics I: Theory of Geophysical Prospection Methods" exercise course
    by Prof. Dr.sc. Florian M. Wagner, Geophysical Imaging and Monitoring (GIM) RWTH Aachen University.
    
    Input:
    data = Seismogram amplitude matrix data over the time. Type: 2-D array.
    t0 = Time at offset=0 or starting time of the trace [s]. Type: float.
    v = Velocity of the curve [m/s]. Type: float.
    time = Time coordinate of one trace [s]. Type: 1-D array with the size of data.shape[0].
    offset = Distance location of each trace [m]. Type: 1-D list or array with the size of data.shape[1].
    
    Output:
    zeros_nmo = Shifted data based on v and t0. Type: 2-D array with the same size of data.
    '''
    zeros_nmo = np.zeros_like(data)
    i = 0
    for trace in data.T:
        nmo = t_nmo(offset[i], t0=t0, v_nmo=v)
        new_nmo = nmo - (t0)
        target_nmo = np.abs(new_nmo - time).argmin()
        zeros_nmo[:,i] = shift(trace, -target_nmo)
        i += 1
    return zeros_nmo

def NMO_correction(data,offset,time,v=4000, t0=0.2):
    '''
    Visualization of the NMO correction based on the shifting.
    This code was modified from "Geophysics I: Theory of Geophysical Prospection Methods" exercise course
    by Prof. Dr.sc. Florian M. Wagner, Geophysical Imaging and Monitoring (GIM) RWTH Aachen University.
    
    This is interactive visualization to play around with the zero-offset time (t0) and the correct
    NMO velocity (v)
    
    Input:
    data = Seismogram amplitude matrix data over the time. Type: 2-D array.
    offset = Distance location of each trace [m]. Type: 1-D list or array with size of data.shape[1].
    time = Time coordinate of one trace [s]. Type: 1-D array with size of data.shape[0].
    t0 = Time at offset=0 or starting time of the trace [s]. Type: float. Default: 0.2.
    v = Velocity of the curve [m/s]. Type: float. Default: 4000.
    
    Output:
    Four matplotlib.pyplot plots with:
    Top left = data with the NMO time plot (to be matched with the curve).
    Top right = Stacked data without applying shift. Always the same.
    Bottom left = Shifted data based on choosen v and t0.
    Bottom right = Stacked data after shifting.
    '''
    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(14,12), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)        
    
    def velocity_test(ax,data,offset,time,v=v, t0=t0):
        nmo = t_nmo(offset, t0=t0, v_nmo=v)
        plot_seismogram(ax, data, offset, time, plot_option=1)
        ax.plot(offset, nmo, c='yellow', linewidth=5)
        
    velocity_test(ax[0,0], data, offset, time)
    
    shifted_nmo = shift_nmo(data, t0, v, time, offset)
        
    plot_seismogram(ax[1,0], shifted_nmo, offset, time, plot_option=1)
    
    ax[0,1].set_title('Trace stacking',fontsize=15,fontweight='bold')
    
    plot_seismogram(ax[0,1], data, offset, time, plot_option=3)
    plot_seismogram(ax[1,1], shifted_nmo, offset, time, plot_option=3)
    
    ax[0,0].set_title('Before NMO',fontsize=15,fontweight='bold')
    ax[1,0].set_title('After NMO',fontsize=15,fontweight='bold')
	
def semblance(data, time, v_range, offset):
    '''
    Calculate semblance as a guidance for velocity analysis. Using formula based on Geldart & Sheriff (2004).
    The highest value will likely shows the correct velocity at certain time.
    
    ***Warning: Running so slow and so far still bad result when tested with the real data***
    
    Input:
    data = Seismogram amplitude matrix data over the time. Type: 2-D array.
    time = Time coordinate of one trace [s]. Type: 1-D array with size of data.shape[0].
    v_range = Range of velocity to be scanned each [m/s]. More is slower, but smoother result. Range from lowest to highest. Type: 1-D list or array.
    offset = Distance location of each trace [m]. Type: 1-D list or array with size of data.shape[1].
    
    Output:
    Semblance = Coherency matrix of the summed trace amplitude based on velocity over the time. Type: 2-D array with the size of time x v_range.
    '''
    Semblance = np.zeros(( len(time), len(v_range) ))
    i = 0
    for T0 in time:
        j = 0
        for vel in v_range:
            semb = shift_nmo(data, round(T0, 2), vel, time, offset)
            Semblance[i,j] = ( (semb[i,:].sum())**2 ) / ( len(semb[i,:])*(semb[i,:]**2).sum() )
            j += 1
        i += 1
    return Semblance

def semblance_analysis(Semblance, data, t0, time, offset, vel_range, v=2000, Veloo=2000, Timee=0.2):
    '''
    Visualization of the semblance result, data seismogram, and the stacked. Interactive button
    and slider should be from different function because so far it still hard to put it here.
    
    Input:
    Semblance = semblance matrix result. Type: 2-D array.
    data = Seismogram amplitude matrix data over the time. Type: 2-D array.
    t0 = widgets.FloatSlider of the zero-offset time [s]. Used in the interactive mode. Type: float.
    time = Time coordinate of one trace [s]. Type: 1-D array with size of data.shape[0].
    offset = Distance location of each trace [m]. Type: 1-D list or array with size of data.shape[1].
    v = widgets.FloatSlider of the NMO velocity [m/s]. Used in the interactive mode. Type: float. Default: 2000.
    Veloo = Should be empty list ([]) in the interactive mode to store the NMO velocity from v. But here declared the default because
            sometimes it's error without it.
    Timee = Should be empty list ([]) in the interactive mode to store the zero-offset time from t0. But here declared the default because
            sometimes it's error without it.
    
    Output:
    Three matplotlib.pyplot plots with:
    Left = Semblance panel to interact and choose the best velocity over the time.
    Center = Seismogram amplitude matrix to be shifted based on the semblance picked.
    Right = Stacked seismogram after shifted based on the semblance picked.
    '''
    Shift_NMO = shift_nmo(data, t0, v, time, offset)
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,10), gridspec_kw={'width_ratios': [2, 3, 1]}, sharey=True)

    ax[0].imshow(Semblance,aspect='auto', cmap='jet',
                   extent=[np.min(vel_range), np.max(vel_range),
                           np.max(time), np.min(time)])
    ax[0].set_title('Semblance', fontsize=14, fontweight='bold')
    ax[0].set_ylabel('Time [s]')
    ax[0].set_xlabel('Velocity [m/s]')
    ax[0].plot(v,t0, 'o', color='w', ms=15, mew=3, lw=10)
    
    if len(Veloo) > 0 and len(Timee) > 0:
        ax[0].plot(Veloo,Timee, '*', color='w', mec='k', mew=2, ms=18, label='Picked location')
        if len(Veloo) > 1 and len(Timee) > 1:
            interpolation = interp1d(Timee, Veloo, kind='linear', fill_value='extrapolate')
            ax[0].plot(interpolation(time), time, color='yellow', label='Interpolated Velocity')
        
    ax[0].vlines(v, np.min(time), np.max(time), colors='w')
    ax[0].hlines(t0, np.min(vel_range), np.max(vel_range), colors='w')
    ax[0].set_xlim(np.min(vel_range), np.max(vel_range))
    ax[0].set_ylim(np.max(time), np.min(time))
    if len(Veloo) == 0 and len(Timee) == 0:
        ax[0].plot([],[],alpha=0,label=' ')
    ax[0].legend(loc='upper right')
    
    plot_seismogram(ax[1], Shift_NMO, offset, time, plot_option=1)
    ax[1].set_title('CMP Gather', fontsize=14, fontweight='bold')
    ax[1].set_ylabel('')

    plot_seismogram(ax[2], Shift_NMO, offset, time, plot_option=3)
    ax[2].set_title('CMP Stack', fontsize=14, fontweight='bold')

    plt.tight_layout()