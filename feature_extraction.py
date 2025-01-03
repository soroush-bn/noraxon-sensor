#todo2
from numpy.typing import NDArray
import numpy as np
from scipy.linalg import toeplitz
from sklearn.feature_selection import mutual_info_classif
from time import time 
import re
class Feature():
    def __init__(self) -> None:
        self.time_domain_features = ["MAV","VAR","rms","WL","DAMV","DASDV","ZC","MYOP","WAMP","SSC","HIST1","HIST2","HIST3","HIST4","HIST5","HIST6","HIST7","HIST8","HIST9","HIST10","AR1","AR2","AR3","AR4"]
        self.frequency_domain_features = ["MNF","MDF","PKF","TTP","SM1","SM2","SM3","FR","PSR","VCF"]
        self.abs_v = None
        self.diff_v = None
        self.abs_diff_v= None
        

        
    def get_time_features(self,v:NDArray):
        if len(v) == 0:
            raise ValueError("Input signal is empty.")
        
    def get_time_features(self,v:NDArray):
        self.abs_v = np.abs(v)
        self.diff_v = np.diff(v)
        self.abs_diff_v = np.abs(self.diff_v)
        
        out = [self.MAV(v),self.VAR(v),self.rms(v),self.WL(v),self.DAMV(v),self.DASDV(v),self.ZC(v),self.MYOP(v),self.WAMP(v),self.SSC(v),*self.HIST(v),*self.AR(v)]
        return out #dict(zip(self.time_domain_features,out))

    def MAV(self,v: NDArray): #v is a window of a gesture repetition    
        return np.mean(self.abs_v)
    
    def VAR(self,v:NDArray):
        return np.var(v)
    
    def rms(self,v:NDArray):
        return np.sqrt(np.mean(v**2))
    
    def WL(self,v:NDArray):
        if v.ndim == 1:  # 1D signal
            wl = np.sum(self.abs_diff_v)
        else:  # ND signal, calculate along axis=0
            wl = np.sum(np.abs(np.diff(v, axis=0)), axis=0)
        return wl

    def DAMV(self,v:NDArray):
        return  np.mean(self.abs_diff_v)
    
    def DASDV(self,v:NDArray):
        return np.std(self.abs_diff_v)
    
    def ZC(self,v:NDArray):
        return len(np.where(np.diff(np.sign(v)))[0])
    
    def MYOP(self,v:NDArray,threshold_ratio=0.1): # how to find this threshhold ? ? ?? ? ?
        threshold = threshold_ratio * np.max(self.abs_v)  # Define the threshold as a ratio of the max absolute value
        if v.ndim == 1:  # 1D signal
            active_samples = np.sum(self.abs_v> threshold)
            mppr = (active_samples / len(v)) * 100
        return mppr
    

    def WAMP(self,v:NDArray,threshold=0.1): # again how to find this threshold
        return np.sum(self.abs_diff_v> threshold)

    def SSC(self,v:NDArray,threshold=0.1): # again
        f = lambda x: (x >= threshold).astype(float)
        SSC = np.sum(f(-np.diff(v,prepend=1)[1:-1]*np.diff(v)[1:]))
        return SSC
    
    def SampEn(self,v:NDArray,m=2,r=None):
        N = len(v)
        if r is None:
            r = 0.2 * np.std(v)  # Default tolerance is 20% of the standard deviation
        
        # Create template vectors of length m and m+1
        def create_templates(signal, m):
            """Create a matrix of subsequences of length m."""
            return np.array([signal[i:i + m] for i in range(N - m + 1)])
        
        def count_similar(template, templates, r):
            """Count the number of templates within a tolerance r."""
            distances = np.max(np.abs(templates - template), axis=1)
            return np.sum(distances <= r) - 1  # Subtract 1 to avoid self-match
        
        templates_m = create_templates(v, m)
        templates_m1 = create_templates(v, m + 1)
        
        B = np.sum([count_similar(templates_m[i], templates_m, r) for i in range(len(templates_m))])
        A = np.sum([count_similar(templates_m1[i], templates_m1, r) for i in range(len(templates_m1))])
        
        # Avoid division by zero (add small epsilon)
        if B == 0 or A == 0:
            return np.inf
        
        sampen = -np.log(A / B)
        return sampen
    

    def HIST(self,v:NDArray,bins=10,normalize = True):
        hist, _ = np.histogram(v, bins=bins, density=normalize)
        return hist


    def AR(self,v:NDArray):
        r = np.correlate(v, v, mode='full') / len(v)  # Autocorrelation
        r = r[len(v) - 1:]  # Use only non-negative lags
        R = toeplitz(r[:4])  # Create Toeplitz matrix
        r_vector = r[1:4 + 1]  # Right-hand side of Yule-Walker equation
        ar_coeffs = np.linalg.solve(R, r_vector)  # Solve R * a = r
        return ar_coeffs
    

    def _MNF(self,power_spectrum,freqs) -> float:
        mnf = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        return mnf
    
    def _MDF(self,power_spectrum,freqs) -> float:
        cumulative_power = np.cumsum(power_spectrum)
        total_power = cumulative_power[-1]
        median_index = np.where(cumulative_power >= total_power / 2)[0][0]
        f_median = freqs[median_index]
        return f_median
    

    def _PKF(self,power_spectrum,freqs) -> float:
        peak_index = np.argmax(power_spectrum)  # Index of max power
        return freqs[peak_index]
    
    def _TTP(self,power_spectrum,freqs) -> float:

        cumulative_power = np.cumsum(power_spectrum)
        total_power = cumulative_power[-1]
        return total_power
    
    def _SM(self,power_spectrum,freqs) -> float:

        f_mean = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
    
        fsm = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        ssm = np.sum(power_spectrum * (freqs - f_mean) ** 2) / np.sum(power_spectrum)
        tsm = np.sum(power_spectrum * (freqs - f_mean) ** 3) / np.sum(power_spectrum)
    
        return fsm,ssm,tsm
    

    def _FSR(self,power_spectrum,freqs) -> float:
        total_power = np.sum(power_spectrum)
        band_indices = np.where((freqs >= 20) & (freqs <= 50))
        band_power = np.sum(power_spectrum[band_indices])
        outside_band_indices = np.where((freqs < 20) | (freqs > 50))
        outside_band_power = np.sum(power_spectrum[outside_band_indices])
        power_spectrum_ratio = band_power / (band_power + outside_band_power)
        frequency_ratio = band_power / total_power
        return frequency_ratio,power_spectrum_ratio
    
    def _VCF(self,power_spectrum,freqs) -> float:
        central_frequency = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        vcf = np.sum(power_spectrum * (freqs - central_frequency) ** 2) / np.sum(power_spectrum)
        return vcf
    
    def get_freq_featrures(self,gesture_signal: np.ndarray,sampling_rate=1000.0):
        # t1= time()
        N = len(gesture_signal)
        fft_values = np.fft.fft(gesture_signal)
        fft_values = np.abs(fft_values[:N // 2])  
        power_spectrum = fft_values ** 2  
        freqs = np.fft.fftfreq(N, d=1 / sampling_rate)[:N // 2]
        out = [self._MNF(power_spectrum,freqs),self._MDF(power_spectrum,freqs),self._PKF(power_spectrum,freqs),self._TTP(power_spectrum,freqs),*self._SM(power_spectrum,freqs),
               *self._FSR(power_spectrum,freqs),self._VCF(power_spectrum,freqs)]
        # t2 = time()
        # print("freq features: ")
        # print(t2-t1)
        return out #dict(zip(self.frequency_domain_features,out))
    
def manual_feature_selector(df,sensor = "*",type= "*", feature="*"    ): 
    columns =[] 
    if sensor=="*":
        s= '[0-8]'
    else:
        ss = sensor.replace(',','|') 
        s= f'({ss})'
    t= '(accel|mag|gyro)' if  type=="*" else type 
    if type=="all":
        t='(accel|mag|gyro|emg)'
    f= '\w' if feature=="*" else feature

            
    imu_columns_pattern = f"sensor{s}_{t}_(x|y|z)_{f}"
    emg_columns_pattern = f"{t}{s}_{f}"
    final_pattern = f"{imu_columns_pattern}|{emg_columns_pattern}"
    p = re.compile(final_pattern)
    all_columns = df.columns
    for c in all_columns:
        r = p.match(c)
        if r != None: 
            columns.append(c)
    
    return columns
        # feature wise:
        # location wise 1-8
        # sensor wise: emg, acc,.. 



def sffs(X, y, max_features=10):
    selected_features = []
    candidate_features = list(range(X.shape[1]))

    while len(selected_features) < max_features:
        best_score = -np.inf
        best_feature = None

        for feature in candidate_features:
            current_features = selected_features + [feature]
            score = mutual_info_classif(X[:, current_features], y, discrete_features=True).mean()
            if score > best_score:
                best_score = score
                best_feature = feature
        
        if best_feature is not None:
            selected_features.append(best_feature)
            candidate_features.remove(best_feature)

        # Step 2: Floating Backward Elimination
        if len(selected_features) > 2:
            worst_score = np.inf
            worst_feature = None
            for feature in selected_features:
                current_features = selected_features.copy()
                current_features.remove(feature)
                score = mutual_info_classif(X[:, current_features], y, discrete_features=True).mean()
                if score < worst_score:
                    worst_score = score
                    worst_feature = feature
            
            if worst_feature is not None and worst_score < best_score:
                selected_features.remove(worst_feature)
                candidate_features.append(worst_feature)

    return selected_features
