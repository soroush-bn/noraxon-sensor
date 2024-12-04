import numpy as np
from scipy.signal import butter, filtfilt, iirnotch,welch
import pandas as pd
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
class Signal:

    def __init__(self, sampling_rate,segment_length):
        self.segment_length = segment_length
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
    def __get_activation_resting(self,channel_data,labels):
        activation_data = []
        rest_data = []

        for d,l in zip(channel_data,labels):
            if l=="rest":
                rest_data.append(d)
            else:
                activation_data.append(d)
        
        return activation_data,rest_data
    
    def FWR(self,forearm_data, wrist_data,labels ):
        forearm_activation, forearm_resting = self.__get_activation_resting(forearm_data,labels)
        wrist_activation, wrist_resting = self.__get_activation_resting(wrist_data,labels)
        

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

        numerator = np.sqrt(rms_forearm_activation**2 - rms_forearm_resting**2)
        denominator = np.sqrt(rms_wrist_activation**2 - rms_wrist_resting**2)
        if denominator == 0:
            raise ValueError("Denominator in FWR calculation is zero, check signal data.")
        fwr = numerator / denominator

        return fwr

    def SNR(self,channel_data):
        activation_unfiltered_data , rest_unfiltered_data = self.__get_activation_resting(channel_data)
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
            raise ValueError("Motion artifacts power is zero. Check the signal data.")
        smr = 10 * np.log10(total_power / motion_artifacts_power)

        return smr


    def omega(self,channel_data):
        freqs, psd = self.compute_psd(channel_data)
        M0, M1, M2 = self.compute_spectral_moments(psd, freqs, max_freq=500)
        if M0 == 0 or M1 == 0:
            raise ValueError("M0 or M1 is zero. Check the signal data for proper PSD computation.")
        omega_value = 10 * np.log10((np.sqrt(M2 / M0)) / (M1 / M0))

        return omega_value
    
    def get_paper_result_for_one_participant_emg(self,participant_final_df):
        df = participant_final_df 
        labels = df["labels"].values
        emg_F1 = df[f"emg{config["F1"]}"].values
        emg_F2 = df[f"emg{config["F2"]}"].values
        emg_F3 = df[f"emg{config["F3"]}"].values
        emg_F4 = df[f"emg{config["F4"]}"].values
        forearm_data =  np.mean([emg_F1, emg_F2, emg_F3, emg_F4], axis=0)
        emg_W1 = df[f"emg{config["W1"]}"].values
        emg_W2 = df[f"emg{config["W2"]}"].values
        emg_W3 = df[f"emg{config["W3"]}"].values
        emg_W4 = df[f"emg{config["W4"]}"].values
        wrist_data =  np.mean([emg_W1, emg_W2, emg_W3, emg_W4], axis=0)
        FWR = self.FWR(forearm_data,wrist_data)

        #Average SNR across different wrist sensors
        SNR_W= self.SNR(np.mean([emg_W1, emg_W2, emg_W3, emg_W4], axis=0))
        #Average SNR across different forearm sensors
        SNR_F= self.SNR(np.mean([emg_F1, emg_F2, emg_F3, emg_F4], axis=0))
        SNR = (SNR_W + SNR_F)/2


        #Average SMR across different wrist sensors
        SMR_W= self.SMR(np.mean([emg_W1, emg_W2, emg_W3, emg_W4], axis=0))
        #Average SMR across different forearm sensors
        SMR_F= self.SMR(np.mean([emg_F1, emg_F2, emg_F3, emg_F4], axis=0))
        SMR = (SMR_W + SMR_F)/2


        #Average Omega across different wrist sensors
        omega_W= self.omega(np.mean([emg_W1, emg_W2, emg_W3, emg_W4], axis=0))
        #Average SMR across different forearm sensors
        omega_F= self.omega(np.mean([emg_F1, emg_F2, emg_F3, emg_F4], axis=0))
        omega = (omega_F + omega_W)/2


        return {"FWR": FWR , "SNR": SNR , "SMR": SMR, "Omega": omega}



        







    
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
    smr = signal_processor.SMR(raw_signal)
    omega = signal_processor.omega(raw_signal)

    # Compute spectral moments
    freqs, psd = signal_processor.compute_psd(raw_signal)
    M0, M1, M2 = signal_processor.compute_spectral_moments(psd, freqs)

    # Print results
    print(f"FWR: {fwr}")
    print(f"SMR: {smr}")
    print(f"Omega: {omega}")
    print(f"Spectral Moments: M0={M0}, M1={M1}, M2={M2}")



if __name__=="__main__":
    main()