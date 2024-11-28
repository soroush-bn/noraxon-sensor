import numpy as np

def downsample(data, k=10):
    # Ensure the data is sorted by time
    data = data.sort_values(by='time')
    
    # Group by chunks of size k and calculate the mean
    downsampled = data.groupby(data.index // k).mean()
    
    # Keep the first time value for each group
    downsampled['time'] = data['time'].iloc[::k].reset_index(drop=True)
    
    # Reorder the columns to place 'time' first
    columns = ['time'] + [col for col in downsampled.columns if col != 'time']
    downsampled = downsampled[columns]
    
    return downsampled

if __name__ == "__main__": 

    pass