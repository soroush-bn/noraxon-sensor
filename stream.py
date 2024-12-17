#this file should give you noraxon streamed data in noraxon.csv file amd first and last time stamps in a text file
import requests
import json
import time
import numpy as np
import yaml
import os
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

directory = os.path.join(config["saving_dir"], f"{config['first_name']}_{config['last_name']}_{config['experiment_name']}")
if directory and not os.path.exists(directory):
    os.makedirs(directory)

def noraxon_stream_init(server_ip='192.168.0.106', port=9220):
    server_url = f"http://{server_ip}:{port}"
    headers_url = f"{server_url}/headers"
    disable_url = f"{server_url}/disable/"
    enable_url = f"{server_url}/enable/"
    first_time =0 
    last_time =0 
    while True:
        try:
            response = requests.get(headers_url)
            data = response.json()

            print("connected sucessfully at time: ")
            first_time = time.time()
            break
        except:
            pass
            # raise Exception(f"Could not connect to MR3 stream at {server_url}. Check if HTTP Streaming is enabled and a measurement is currently running.")
        
    if len(data['headers']) == 0:
        raise Exception("No data sources found")
    
    sources = []
    for header in data['headers']:
        type_ = header['type'].replace('real.', '').replace('vector3.accel', 'acceleration').replace('vector3.rot', 'quaternion').replace('vector3.pos', 'position').replace('switch', 'on/off')
        sources.append(f"{header['name']} [{type_}]")
    
    sel = np.arange(79).tolist()#input_selection_dialog(sources)
    if not sel:
        raise Exception("No data sources selected")
    
    stream_config = {'server_url': server_url, 'channels': []}
    for index in sel:
        header = data['headers'][index]
        type_ = header['type'].replace('real.', '').replace('vector3.accel', 'acceleration').replace('vector3.rot', 'quaternion').replace('vector3.pos', 'position').replace('switch', 'on/off')
        stream_config['channels'].append({
            'name': header['name'],
            'type': type_,
            'full_type': header['type'],
            'sample_rate': header['samplerate'],
            'units': header['units'],
            'index': header['index']
        })
        
        try:
            response = requests.get(f"{enable_url}{header['index']}")
        except:
            raise Exception(f"Could not connect to MR3 stream at {server_url}. Check if HTTP Streaming is enabled and a measurement is currently running.")
    
    return stream_config

def noraxon_stream_collect(stream_config, seconds=10):
    data = []
    for channel in stream_config['channels']:
        print (f"channels{channel}")
        data.append({'info': channel, 'samples': []})
    print(seconds)
    samples_remaining = [channel['sample_rate'] * seconds for channel in stream_config['channels']]
    last_data_timer = time.time()
    first_time = time.time()
    print("total samples")
    print(samples_remaining)
    while max(samples_remaining) > 0:   
        print(max(samples_remaining))
        try:
            response = requests.get(f"{stream_config['server_url']}/samples")
            new_data = response.json()
        except:
            new_data = None
        
        if new_data:
            last_data_timer = time.time()
            
            for channel_data in new_data['channels']:
                for i, channel in enumerate(data):
                    if channel['info']['index'] == channel_data['index']:
                        to_copy = min(len(channel_data['samples']), samples_remaining[i])
                        if 'vector3.rot' in channel['info']['full_type']:
                            samples = []
                            for j in range(0, to_copy, 3):
                                tmpx, tmpy, tmpz = channel_data['samples'][j:j+3]
                                q0 = (max(0, 1 - (tmpx**2 + tmpy**2 + tmpz**2)))**0.5
                                samples.append({'q0': q0, 'q1': tmpx, 'q2': tmpy, 'q3': tmpz})
                            channel['samples'].extend(samples)
                        elif 'vector3' in channel['info']['full_type']:
                            samples = [{'x': channel_data['samples'][j], 'y': channel_data['samples'][j+1], 'z': channel_data['samples'][j+2]} for j in range(0, to_copy, 3)]
                            channel['samples'].extend(samples)
                        else:
                            channel['samples'].extend(channel_data['samples'][:to_copy])
                        samples_remaining[i] -= to_copy
                        break
        else:
            if time.time() - last_data_timer > 10:
                # raise Exception("MR3 stream timeout. Check if HTTP Streaming is enabled and a measurement is currently running.")
                pass
    last_time = time.time()
    return data,first_time,last_time



def input_selection_dialog(sources):
    print("Select stream data:")
    for i, source in enumerate(sources):
        print(f"{i+1}. {source}")
    
    sel = input("Enter comma-separated indices (e.g., 1,3,4): ")
    try:
        indices = [int(idx) - 1 for idx in sel.split(",")]
        return indices
    except:
        return None
import csv

# Function to save collected data to a CSV file
import csv

def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(['Channel Name', 'Type', 'Sample Rate', 'Units', 'Index', 'Sample'])
        
        # Write data rows
        for channel_data in data:
            channel_info = channel_data['info']
            channel_name = channel_info['name']
            channel_type = channel_info['type']
            sample_rate = channel_info['sample_rate']
            units = channel_info['units']
            index = channel_info['index']
            # time_stamp = channel_info["time"]
            for sample in channel_data['samples']:
                # Write each sample as a separate row
                writer.writerow([channel_name, channel_type, sample_rate, units, index, sample])


if __name__=="__main__":
    duration = (config["gesture_duration"] + config["rest_duration"]) *config["number_of_gesture_repetition"] *config["number_of_gestures"] 
    print(duration)
    stream_config  = noraxon_stream_init('127.0.0.1', 9220)
    data,first,last = noraxon_stream_collect(stream_config, duration)
    save_at =  os.path.join(directory, 'noraxon.csv')
    save_to_csv(data, save_at)
    
    text_dir = os.path.join(directory,"timestamps.txt")
    with open(text_dir,'w') as f:
        f.write(str(first))
        f.write(",")
        f.write(str(last))
        f.write(",")
        f.write(str(duration))
    print("Data collected successfully!")