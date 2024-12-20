import numpy as np
from scipy.signal import butter, filtfilt, iirnotch,welch
import pandas as pd
import yaml
from scipy.stats import f_oneway, ttest_ind

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
class Signal:

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate


    def calculate_p_value(self, group1,group2):

        t_stat, p_value = ttest_ind(group1, group2)
        return p_value

    def calculate_cohens_d(self, group1, group2):
        """
        Calculate Cohen's d to measure the effect size between two groups.
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        d_value = (mean1 - mean2) / pooled_std
        return d_value
    

    def butter_bandpass_filter(self, data, lowcut, highcut, order=3):
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def notch_filter(self, data, freq=60.0, Q=50.0):
        nyquist = 0.5 * self.sampling_rate
        w0 = freq / nyquist
        b, a = iirnotch(w0, Q)
        return filtfilt(b, a, data)


    def get_signals_data(self,df):
        signals= []
        return signals
    

    def compute_psd(self, signal, nperseg=1024):
        freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=nperseg)
        return freqs, psd
    
    def linear_fit_psd(self, freqs, psd):
        max_power_idx = np.argmax(psd)
        max_freq = freqs[max_power_idx]
        slope = psd[max_power_idx] / max_freq
        linear_psd = slope * freqs
        return linear_psd
    
    def rms(self, data):
        return np.sqrt(np.mean(np.square(data)))
    
    def compute_spectral_moments(self, psd, freqs, max_freq=500):
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
    
    def ___get_activation_resting_IMU(self,df,type="accel"):
        active_df =  df[df["label"] != "rest"]
        rest_df = df[df["label"] == "rest"]
        active = []
        rest= []  

        forearm_activation= active_df[[
        f'sensor{config["F1"]}_{type}_x', 
        f'sensor{config["F2"]}_{type}_x', 
        f'sensor{config["F3"]}_{type}_x', 
        f'sensor{config["F4"]}_{type}_x',
                f'sensor{config["F1"]}_{type}_y', 
        f'sensor{config["F2"]}_{type}_y', 
        f'sensor{config["F3"]}_{type}_y', 
        f'sensor{config["F4"]}_{type}_y',
                f'sensor{config["F1"]}_{type}_z', 
        f'sensor{config["F2"]}_{type}_z', 
        f'sensor{config["F3"]}_{type}_z', 
        f'sensor{config["F4"]}_{type}_z',
        ]].mean(axis=1)

        forearm_resting= rest_df[[
        f'sensor{config["F1"]}_{type}_x', 
        f'sensor{config["F2"]}_{type}_x', 
        f'sensor{config["F3"]}_{type}_x', 
        f'sensor{config["F4"]}_{type}_x',
                f'sensor{config["F1"]}_{type}_y', 
        f'sensor{config["F2"]}_{type}_y', 
        f'sensor{config["F3"]}_{type}_y', 
        f'sensor{config["F4"]}_{type}_y',
                f'sensor{config["F1"]}_{type}_z', 
        f'sensor{config["F2"]}_{type}_z', 
        f'sensor{config["F3"]}_{type}_z', 
        f'sensor{config["F4"]}_{type}_z',
        ]].mean(axis=1)

        wrist_activation= active_df[[
        f'sensor{config["W1"]}_{type}_x', 
        f'sensor{config["W2"]}_{type}_x', 
        f'sensor{config["W3"]}_{type}_x', 
        f'sensor{config["W4"]}_{type}_x',
                f'sensor{config["W1"]}_{type}_y', 
        f'sensor{config["W2"]}_{type}_y', 
        f'sensor{config["W3"]}_{type}_y', 
        f'sensor{config["W4"]}_{type}_y',
                f'sensor{config["W1"]}_{type}_z', 
        f'sensor{config["W2"]}_{type}_z', 
        f'sensor{config["W3"]}_{type}_z', 
        f'sensor{config["W4"]}_{type}_z',
        ]].mean(axis=1)

        wrist_resting= rest_df[[
        f'sensor{config["W1"]}_{type}_x', 
        f'sensor{config["W2"]}_{type}_x', 
        f'sensor{config["W3"]}_{type}_x', 
        f'sensor{config["W4"]}_{type}_x',
                f'sensor{config["W1"]}_{type}_y', 
        f'sensor{config["W2"]}_{type}_y', 
        f'sensor{config["W3"]}_{type}_y', 
        f'sensor{config["W4"]}_{type}_y',
                f'sensor{config["W1"]}_{type}_z', 
        f'sensor{config["W2"]}_{type}_z', 
        f'sensor{config["W3"]}_{type}_z', 
        f'sensor{config["W4"]}_{type}_z',
        ]].mean(axis=1)

        forearm_raw= df[[
        f'sensor{config["F1"]}_{type}_x', 
        f'sensor{config["F2"]}_{type}_x', 
        f'sensor{config["F3"]}_{type}_x', 
        f'sensor{config["F4"]}_{type}_x',
                f'sensor{config["F1"]}_{type}_y', 
        f'sensor{config["F2"]}_{type}_y', 
        f'sensor{config["F3"]}_{type}_y', 
        f'sensor{config["F4"]}_{type}_y',
                f'sensor{config["F1"]}_{type}_z', 
        f'sensor{config["F2"]}_{type}_z', 
        f'sensor{config["F3"]}_{type}_z', 
        f'sensor{config["F4"]}_{type}_z',
        ]].mean(axis=1)

        forearm_raw= df[[
        f'sensor{config["F1"]}_{type}_x', 
        f'sensor{config["F2"]}_{type}_x', 
        f'sensor{config["F3"]}_{type}_x', 
        f'sensor{config["F4"]}_{type}_x',
                f'sensor{config["F1"]}_{type}_y', 
        f'sensor{config["F2"]}_{type}_y', 
        f'sensor{config["F3"]}_{type}_y', 
        f'sensor{config["F4"]}_{type}_y',
                f'sensor{config["F1"]}_{type}_z', 
        f'sensor{config["F2"]}_{type}_z', 
        f'sensor{config["F3"]}_{type}_z', 
        f'sensor{config["F4"]}_{type}_z',
        ]].mean(axis=1)


        wrist_raw= df[[
        f'sensor{config["W1"]}_{type}_x', 
        f'sensor{config["W2"]}_{type}_x', 
        f'sensor{config["W3"]}_{type}_x', 
        f'sensor{config["W4"]}_{type}_x',
                f'sensor{config["W1"]}_{type}_y', 
        f'sensor{config["W2"]}_{type}_y', 
        f'sensor{config["W3"]}_{type}_y', 
        f'sensor{config["W4"]}_{type}_y',
                f'sensor{config["W1"]}_{type}_z', 
        f'sensor{config["W2"]}_{type}_z', 
        f'sensor{config["W3"]}_{type}_z', 
        f'sensor{config["W4"]}_{type}_z',
        ]].mean(axis=1)

        wrist_raw= df[[
        f'sensor{config["W1"]}_{type}_x', 
        f'sensor{config["W2"]}_{type}_x', 
        f'sensor{config["W3"]}_{type}_x', 
        f'sensor{config["W4"]}_{type}_x',
                f'sensor{config["W1"]}_{type}_y', 
        f'sensor{config["W2"]}_{type}_y', 
        f'sensor{config["W3"]}_{type}_y', 
        f'sensor{config["W4"]}_{type}_y',
                f'sensor{config["W1"]}_{type}_z', 
        f'sensor{config["W2"]}_{type}_z', 
        f'sensor{config["W3"]}_{type}_z', 
        f'sensor{config["W4"]}_{type}_z',
        ]].mean(axis=1)
        return forearm_activation,forearm_resting,wrist_activation,wrist_resting,forearm_raw,wrist_raw

        


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

        raw_signal_forearm = df[[
            f'emg{config["F1"]}', 
            f'emg{config["F2"]}', 
            f'emg{config["F3"]}', 
            f'emg{config["F4"]}'
        ]].mean(axis=1)
        
        raw_signal_wrist = df[[
            f'emg{config["W1"]}', 
            f'emg{config["W2"]}', 
            f'emg{config["W3"]}', 
            f'emg{config["W4"]}'
        ]].mean(axis=1)
        
        return forearm_activation,forearm_resting,wrist_activation,wrist_resting,raw_signal_forearm,raw_signal_wrist
    
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

    def SNR(self,activation,resting):
        activation_unfiltered_data = activation#np.mean([forearm_activation,wrist_activation],axis=0)
        rest_unfiltered_data = resting #np.mean([forearm_resting,wrist_resting],axis=0)

        activation_filtered_data = self.notch_filter(activation_unfiltered_data)
        activation_filtered_data = self.butter_bandpass_filter(activation_filtered_data,20,500) #change

        return 20*np.log10(self.rms(activation_filtered_data)/self.rms(rest_unfiltered_data))
    
    
    
    def SMR(self, channel_data):
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
            forearm_activation,forearm_resting,wrist_activation,wrist_resting,raw_signal_forearm,raw_signal_wrist= self.__get_activation_resting(gdf)
            forearm_activation_acc,forearm_resting_acc,wrist_activation_acc,wrist_resting_acc,raw_signal_forearm_acc,raw_signal_wrist_acc= self.___get_activation_resting_IMU(gdf,"accel")
            forearm_activation_gyro,forearm_resting_gyro,wrist_activation_gyro,wrist_resting_gyro,raw_signal_forearm_gyro,raw_signal_wrist_gyro= self.___get_activation_resting_IMU(gdf,"gyro")
            forearm_activation_mag,forearm_resting_mag,wrist_activation_mag,wrist_resting_mag,raw_signal_forearm_mag,raw_signal_wrist_mag= self.___get_activation_resting_IMU(gdf,"mag")

            
            fwr = self.FWR(forearm_activation, forearm_resting, wrist_activation, wrist_resting)
            fwr_accel = self.FWR(forearm_activation_acc, forearm_resting_acc, wrist_activation_acc, wrist_resting_acc)
            fwr_gyro = self.FWR(forearm_activation_gyro, forearm_resting_gyro, wrist_activation_gyro, wrist_resting_gyro)
            fwr_mag = self.FWR(forearm_activation_mag, forearm_resting_mag, wrist_activation_mag, wrist_resting_mag)


            #todo: these values should be calculated after getting participants groups            
            # p_value_fwr = self.calculate_p_value(fwr, 1)
            # d_value_fwr = self.calculate_cohens_d(fwr, 1)

            snr_wrist = self.SNR( wrist_activation, wrist_resting)
            snr_forearm = self.SNR( forearm_activation,forearm_resting)
            snr_wrist_accel = self.SNR( wrist_activation_acc,wrist_resting_acc)
            snr_forearm_accel = self.SNR( forearm_activation_acc,forearm_resting_acc)
            snr_wrist_gyro = self.SNR( wrist_activation_gyro,wrist_resting_gyro)
            snr_forearm_gyro = self.SNR( forearm_activation_gyro,forearm_resting_gyro)
            snr_wrist_mag = self.SNR( wrist_activation_mag,wrist_resting_mag)
            snr_forearm_mag = self.SNR( forearm_activation_mag,forearm_resting_mag)
            # p_value_snr = self.calculate_p_value(snr_wrist,snr_forearm)
            # d_value_snr= self.calculate_cohens_d(snr_wrist,snr_forearm)

            smr_forearm = self.SMR(raw_signal_forearm)
            smr_wrist= self.SMR(raw_signal_wrist)
            smr_forearm_accel = self.SMR(raw_signal_forearm_acc)
            smr_wrist_accel= self.SMR(raw_signal_wrist_acc)
            smr_forearm_gyro = self.SMR(raw_signal_forearm_gyro)
            smr_wrist_gyro= self.SMR(raw_signal_wrist_gyro)
            smr_forearm_mag = self.SMR(raw_signal_forearm_mag)
            smr_wrist_mag= self.SMR(raw_signal_wrist_mag)


            # p_value_smr = self.calculate_p_value(smr_wrist,smr_forearm)
            # d_value_smr= self.calculate_cohens_d(smr_wrist,smr_forearm)

            omega_forearm = self.omega(raw_signal_forearm)
            omega_wrist = self.omega(raw_signal_wrist)
            omega_forearm_accel = self.omega(raw_signal_forearm_acc)
            omega_wrist_accel = self.omega(raw_signal_wrist_acc)
            omega_forearm_gyro = self.omega(raw_signal_forearm_gyro)
            omega_wrist_gyro = self.omega(raw_signal_wrist_gyro)
            omega_forearm_mag = self.omega(raw_signal_forearm_mag)
            omega_wrist_mag = self.omega(raw_signal_wrist_mag)
            # p_value_omega = self.calculate_p_value(omega_wrist,omega_forearm)
            # d_value_omega= self.calculate_cohens_d(omega_wrist,omega_forearm)
            

            print("--"*10)
            print("EMG Measures for label = " + str(label))
            print(f"FWR: {fwr} | P-Value:  | Cohen's d: ")
            print(f"SNR Wrist: {snr_wrist} | SNR Forearm: {snr_forearm} | P-Value:  | Cohen's d: ")
            print(f"SMR Wrist: {smr_wrist} | SMR Forearm: {smr_forearm} | P-Value:  | Cohen's d: ")
            print(f"Omega Wrist: {omega_wrist} | Omega Forearm: {omega_forearm} | P-Value: | Cohen's d:")
            print("--"*10)

            print("--"*10)
            print("IMU accel Measures for label = " + str(label))
            print(f"FWR: {fwr_accel} | P-Value:  | Cohen's d: ")
            print(f"SNR Wrist: {snr_wrist_accel} | SNR Forearm: {snr_forearm_accel} | P-Value:  | Cohen's d: ")
            print(f"SMR Wrist: {smr_wrist_accel} | SMR Forearm: {smr_forearm_accel} | P-Value:  | Cohen's d: ")
            print(f"Omega Wrist: {omega_wrist_accel} | Omega Forearm: {omega_forearm_accel} | P-Value: | Cohen's d:")
            print("--"*10)

            
            print("--"*10)
            print("IMU gyro Measures for label = " + str(label))
            print(f"FWR: {fwr_gyro} | P-Value:  | Cohen's d: ")
            print(f"SNR Wrist: {snr_wrist_gyro} | SNR Forearm: {snr_forearm_gyro} | P-Value:  | Cohen's d: ")
            print(f"SMR Wrist: {smr_wrist_gyro} | SMR Forearm: {smr_forearm_gyro} | P-Value:  | Cohen's d: ")
            print(f"Omega Wrist: {omega_wrist_gyro} | Omega Forearm: {omega_forearm_gyro} | P-Value: | Cohen's d:")
            print("--"*10)


                        
            print("--"*10)
            print("IMU mag Measures for label = " + str(label))
            print(f"FWR: {fwr_mag} | P-Value:  | Cohen's d: ")
            print(f"SNR Wrist: {snr_wrist_mag} | SNR Forearm: {snr_forearm_mag} | P-Value:  | Cohen's d: ")
            print(f"SMR Wrist: {smr_wrist_mag} | SMR Forearm: {smr_forearm_mag} | P-Value:  | Cohen's d: ")
            print(f"Omega Wrist: {omega_wrist_mag} | Omega Forearm: {omega_forearm_mag} | P-Value: | Cohen's d:")
            print("--"*10)

            #todo Qs: are these metrics correct for IMU? 



    
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