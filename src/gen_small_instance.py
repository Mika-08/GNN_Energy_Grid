import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import toml
import torch
from torch_geometric.utils import from_networkx
import json
import os


ORIGINAL_PATH = {
    "transmission_path": "case_studies/stylized_EU/inputs/transmission_lines.csv",
    "demand_path" : "case_studies/stylized_EU/inputs/demand.csv",
    "generation_availability_path" : "case_studies/stylized_EU/inputs/generation_availability.csv",
    "generation_path" : "case_studies/stylized_EU/inputs/generation.csv",
    "scalars_path": "case_studies/stylized_EU/inputs/scalars.toml" 

}




def read_data(ORIGINAL_PATH):
    transmission_df = pd.read_csv(ORIGINAL_PATH['transmission_path'])
    demand_df = pd.read_csv(ORIGINAL_PATH['demand_path'])
    generation_availability_df = pd.read_csv(ORIGINAL_PATH['generation_availability_path'])  
    generation_df = pd.read_csv(ORIGINAL_PATH['generation_path'])

    with open(ORIGINAL_PATH['scalars_path'], 'r') as f:
        scalars = toml.load(f)

    value_of_lost_load = scalars['value_of_lost_load']
    relaxation = scalars['relaxation']

    return transmission_df, demand_df, generation_availability_df, generation_df, value_of_lost_load, relaxation


def generate_small_instances(kept_countries, kept_technologies, kept_time_steps_lb, kept_time_steps_ub, out_folder):
    """
    Generate a small instance of the grid graph with only the specified countries and technologies.
    """
    # Read the original data
    transmission_df, demand_df, generation_availability_df, generation_df, value_of_lost_load, relaxation = read_data(
        ORIGINAL_PATH
    )

    new_transmission_df = transmission_df[transmission_df['from'].isin(kept_countries) & transmission_df['to'].isin(kept_countries)]

    # Filter the dataframes based on the kept countries and technologies
    new_demand_df = demand_df[demand_df['location'].isin(kept_countries)]
    new_generation_df = generation_df[generation_df['location'].isin(kept_countries) & generation_df['technology'].isin(kept_technologies)]
    new_generation_availability_df = generation_availability_df[generation_availability_df['location'].isin(kept_countries) &
                                                                generation_availability_df['technology'].isin(kept_technologies)]

    print(f"After filtering by country: Demand shape: {new_demand_df.shape}, Generation availability shape: {new_generation_availability_df.shape}, Generation shape: {new_generation_df.shape}")

    # Keep only the specified time steps
    new_demand_df = new_demand_df[(new_demand_df['time_step'] >= kept_time_steps_lb) & 
                                (new_demand_df['time_step'] <= kept_time_steps_ub)]

    new_generation_availability_df = new_generation_availability_df[(new_generation_availability_df['time_step'] >= kept_time_steps_lb) & 
                                                                    (new_generation_availability_df['time_step'] <= kept_time_steps_ub)]


    print(f"After filtering by time steps: Demand shape: {new_demand_df.shape}, Generation availability shape: {new_generation_availability_df.shape}, Generation shape: {new_generation_df.shape}")

    # Save the filtered dataframes to new CSV files
    # Ensure the output folder exists
    os.makedirs(out_folder, exist_ok=True)

    new_transmission_df.to_csv(f"{out_folder}/transmission_lines.csv", index=False)
    new_demand_df.to_csv(f"{out_folder}/demand.csv", index=False)
    new_generation_availability_df.to_csv(f"{out_folder}/generation_availability.csv", index=False)
    new_generation_df.to_csv(f"{out_folder}/generation.csv", index=False)
    with open(f"{out_folder}/scalars.toml", 'w') as f:
        toml.dump({"value_of_lost_load": value_of_lost_load, "relaxation": relaxation}, f)


if __name__ == "__main__":
    
    '''
    Technologies: ['Coal' 'Gas' 'Lignite' 'Nuclear' 'Oil' 'SunPV' 'WindOff' 'WindOn']
    Countries: ['AUS' 'BEL' 'BLK' 'BLT' 'CZE' 'DEN' 'FIN' 'FRA' 'GER' 'IRE' 'ITA' 'NED'
    'NOR' 'POL' 'POR' 'SKO' 'SPA' 'SWE' 'SWI' 'UKI']

    Time steps: 1... 8750

    '''
    OUT_FOLDER = "small_instance5-5000/inputs"
    kept_countries = ['GER','FRA','AUS']
    kept_technologies = ['SunPV','Oil', 'WindOn']
    kept_time_steps_lb = 1
    kept_time_steps_ub = 5000

    # Generate the small instance
    generate_small_instances(kept_countries, kept_technologies, kept_time_steps_lb, kept_time_steps_ub, out_folder = OUT_FOLDER)