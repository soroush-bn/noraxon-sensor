import pandas as pd
import yaml
import os 


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
# directory = config["saving_dir"] +'/'+ config["first_name"]+'_' + config['last_name']+'_' + config["experiment_name"] 
directory = config["saving_dir"] +'/'+ config["first_name"]+'_' + config['last_name']+'_' + config["experiment_name"] 
print
# directory = os.path.dirname()
if directory and not os.path.exists(directory):
    os.makedirs(directory)


df = pd.read_csv(r"E:\projects\noraxon\noraxon-sensor\data\aliso_Baghernezhad_second\final_df.csv")
print(df.head())