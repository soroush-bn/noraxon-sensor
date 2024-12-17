import numpy as np
from scipy.signal import butter, filtfilt, iirnotch,welch
import pandas as pd
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
class Signal:

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def butter_bandpass_filter(self, data, lowcut, highcut, order=3):
        """
        Apply a Butterworth bandpass filter to the data.
        
        :param data: Signal data to filter.
        :param lowcut: Low cutoff frequency in Hz.
        :param highcut: High cutoff frequency in Hz.
        :param order: Order of the filter.
        :return: Filtered signal.
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def notch_filter(self, data, freq=60.0, Q=50.0):
        """
        Apply a notch filter to remove interference at the given frequency.
        
        :param data: Signal data to filter.
        :param freq: Frequency to notch (default 60 Hz).
        :param Q: Quality factor of the notch filter.
        :return: Filtered signal.
        """
        nyquist = 0.5 * self.sampling_rate
        w0 = freq / nyquist
        b, a = iirnotch(w0, Q)
        return filtfilt(b, a, data)


    def get_signals_data(self,df):
        signals= []
        return signals
    

    def compute_psd(self, signal, nperseg=1024):
        """
        Compute the Power Spectral Density (PSD) of a signal using Welch's method.
        
        :param signal: Input signal.
        :param nperseg: Length of each segment for Welch's method.
        :return: Frequencies and PSD values.
        """
        freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=nperseg)
        return freqs, psd
    
    def linear_fit_psd(self, freqs, psd):
        """
        Fit a straight line to the PSD from 0 Hz to the highest mean power point.
        
        :param freqs: Frequency values of the PSD.
        :param psd: PSD values.
        :return: PSD values along the linear fit line.
        """
        max_power_idx = np.argmax(psd)
        max_freq = freqs[max_power_idx]
        slope = psd[max_power_idx] / max_freq
        linear_psd = slope * freqs
        return linear_psd
    
    def rms(self, data):
        return np.sqrt(np.mean(np.square(data)))
    def compute_spectral_moments(self, psd, freqs, max_freq=500):
        """
        Compute spectral moments M0, M1, and M2.
        
        :param psd: Power Spectral Density values.
        :param freqs: Frequency values corresponding to the PSD.
        :param max_freq: Maximum frequency to consider for the calculation (e.g., 500 Hz).
        :return: M0, M1, and M2 spectral moments.
        """
        psd_filtered = psd[(freqs >= 0) & (freqs <= max_freq)]
        freqs_filtered = freqs[(freqs >= 0) & (freqs <= max_freq)]

        M0 = np.sum(psd_filtered)
        M1 = np.sum(psd_filtered * freqs_filtered)
        M2 = np.sum(psd_filtered * (freqs_filtered**2))

        return M0, M1, M2
    
    def get_gestures_dataframes(self,df):
        dataframes = []
        temp = df.loc[0,"label"]
        pre_i=0
        for i in range(len(df)):
            if df.loc[i,"label"]!=temp and df.loc[i,"label"]!="rest": 
                dataframes.append(df.iloc[pre_i:i])
                # print(len(df.iloc[pre_i:i]))

                pre_i=i
                temp = df.loc[i,"label"]
        # if len(dataframes)!= config["number_of_gestures"]-1: raise Exception(f' incomplete data {config["number_of_gestures"]} vs {len(dataframes)}')
        return dataframes
    def __get_activation_resting(self,df):
        active_df =  df[df["label"] != "rest"]
        rest_df = df[df["label"] == "rest"]
        forearm_activation = active_df[[
        f'emg{config["F1"]}', 
        f'emg{config["F2"]}', 
        f'emg{config["F3"]}', 
        f'emg{config["F4"]}'
        ]].mean(axis=1)

        forearm_resting = rest_df[[
            f'emg{config["F1"]}', 
            f'emg{config["F2"]}', 
            f'emg{config["F3"]}', 
            f'emg{config["F4"]}'
        ]].mean(axis=1)

        wrist_activation = active_df[[
            f'emg{config["W1"]}', 
            f'emg{config["W2"]}', 
            f'emg{config["W3"]}', 
            f'emg{config["W4"]}'
        ]].mean(axis=1)

        wrist_resting = rest_df[[
            f'emg{config["W1"]}', 
            f'emg{config["W2"]}', 
            f'emg{config["W3"]}', 
            f'emg{config["W4"]}'
        ]].mean(axis=1)

        raw_signal = df[[
            f'emg{config["F1"]}', 
            f'emg{config["F2"]}', 
            f'emg{config["F3"]}', 
            f'emg{config["F4"]}'
        ]].mean(axis=1)
        
        return forearm_activation,forearm_resting,wrist_activation,wrist_resting,raw_signal
    
    def FWR(self,forearm_activation,forearm_resting, wrist_activation,wrist_resting ):
        # forearm_activation, forearm_resting = self.__get_activation_resting(forearm_data,labels)
        # wrist_activation, wrist_resting = self.__get_activation_resting(wrist_data,labels)

        forearm_activation = self.notch_filter(forearm_activation)
        forearm_resting = self.notch_filter(forearm_resting)
        wrist_activation = self.notch_filter(wrist_activation)
        wrist_resting = self.notch_filter(wrist_resting)

        forearm_activation = self.butter_bandpass_filter(forearm_activation, 20, 500)
        forearm_resting = self.butter_bandpass_filter(forearm_resting, 20, 500)
        wrist_activation = self.butter_bandpass_filter(wrist_activation, 20, 500)
        wrist_resting = self.butter_bandpass_filter(wrist_resting, 20, 500)

        rms_forearm_activation = self.rms(forearm_activation)
        rms_forearm_resting = self.rms(forearm_resting)
        rms_wrist_activation = self.rms(wrist_activation)
        rms_wrist_resting = self.rms(wrist_resting)

        numerator = np.sqrt(np.abs(rms_forearm_activation**2 - rms_forearm_resting**2))
        denominator = np.sqrt(np.abs(rms_wrist_activation**2 - rms_wrist_resting**2))
        if denominator == 0:
            print( ValueError("Denominator in FWR calculation is zero, check signal data."))
            fwr =0 
        else:

            fwr = numerator / denominator

        return fwr

    def SNR(self,forearm_activation,forearm_resting, wrist_activation,wrist_resting):
        activation_unfiltered_data = np.mean([forearm_activation,wrist_activation],axis=0)
        rest_unfiltered_data = np.mean([forearm_resting,wrist_resting],axis=0)

        activation_filtered_data = self.notch_filter(activation_unfiltered_data)
        activation_filtered_data = self.butter_bandpass_filter(activation_filtered_data,20,500) #change

        return 20*np.log10(self.rms(activation_filtered_data)/self.rms(rest_unfiltered_data))
    
    
    
    def SMR(self, channel_data):
        """
        Calculate the Signal-to-Motion Artifact Ratio (SMR).

        :param raw_signal: Input raw signal.
        :return: SMR value.
        """
        # Compute the PSD of the raw signal
        freqs, psd = self.compute_psd(channel_data)

        psd_filtered = psd[(freqs >= 0) & (freqs <= 500)]
        freqs_filtered = freqs[(freqs >= 0) & (freqs <= 500)]

        linear_psd = self.linear_fit_psd(freqs_filtered, psd_filtered)

        psd_motion_artifacts = psd_filtered[freqs_filtered <= 20] - linear_psd[freqs_filtered <= 20]
        psd_motion_artifacts = np.maximum(psd_motion_artifacts, 0) 

        total_power = np.sum(psd_filtered)
        motion_artifacts_power = np.sum(psd_motion_artifacts)

        if motion_artifacts_power == 0:
            # raise ValueError("Motion artifacts power is zero. Check the signal data.")
            smr =0
        else:

            smr = 10 * np.log10(total_power / motion_artifacts_power)

        return smr


    def omega(self,channel_data):
        freqs, psd = self.compute_psd(channel_data)
        M0, M1, M2 = self.compute_spectral_moments(psd, freqs, max_freq=500)
        if M0 == 0 or M1 == 0:
            # raise ValueError("M0 or M1 is zero. Check the signal data for proper PSD computation.")
            omega_value=0
        else:

            omega_value = 10 * np.log10((np.sqrt(M2 / M0)) / (M1 / M0))

        return omega_value
        
    def calculate_per_gesture(self,dfs):
        print(len(dfs))
        for gdf in dfs: 
            
            label = gdf["label"].head(1).values
            forearm_activation,forearm_resting,wrist_activation,wrist_resting,raw_signal= self.__get_activation_resting(gdf)
            fwr = self.FWR(forearm_activation, forearm_resting, wrist_activation, wrist_resting)
            snr = self.SNR(forearm_activation, forearm_resting, wrist_activation, wrist_resting)
            smr = self.SMR(raw_signal)
            omega = self.omega(raw_signal)

            print("measures for label =  " + str(label))
            print(f"FWR: {fwr}")
            print(f"SNR: {snr}")

            print(f"SMR: {smr}")
            print(f"Omega: {omega}")
            



    
def main():
    # Load the CSV file
    df = pd.read_csv(r"E:\projects\noraxon\noraxon-sensor\data\aliso_Baghernezhad_second\final_df.csv")


    # Initialize Signal class with the sampling rate (e.g., 2000 Hz)
    sampling_rate = 2000
    signal_processor = Signal(sampling_rate,sampling_rate*225/1000)

    # Extract signals from the DataFrame
    forearm_activation = df['sensor1_accel_x'].values
    forearm_resting = df['sensor1_accel_y'].values
    wrist_activation = df['sensor1_accel_z'].values
    wrist_resting = df['emg1'].values
    raw_signal = df['sensor1_accel_x'].values  # Use this for SMR and Omega

    # Compute metrics
    fwr = signal_processor.FWR(forearm_activation, forearm_resting, wrist_activation, wrist_resting)
    snr = signal_processor.SNR(forearm_activation, forearm_resting, wrist_activation, wrist_resting)
    smr = signal_processor.SMR(raw_signal)
    omega = signal_processor.omega(raw_signal)

    # Print results
    print(f"FWR: {fwr}")
    print(f"SNR: {snr}")

    print(f"SMR: {smr}")
    print(f"Omega: {omega}")


if __name__=="__main__":
    main()