import os
from GraphBuilder import read_input_data, read_output_data
import Report
import pandas as pd

'''
Function below is used to generate summary code for the solver
'''

def generate_summary_data_solver(base_path):
    lines_df, demand_df, generation_availability_df, generation_df, value_of_lost_load, relaxation = read_input_data(base_path)
    investment_df, line_flow_df, loss_of_load_df, production_df, scalar = read_output_data(base_path)

    # Give the shape of demand_df
    T = demand_df['time_step'].nunique()
    print(f"Total time step {T}")
    individual_runtime = scalar['runtime'] / T
    print(f"Individual runtime: {individual_runtime:.3} seconds per time step")


    # Cost calculation
    # report = Report(loss_cost = value_of_lost_load)

    # Prod*generation_cost + V_loss*loss_prod
    print(f"Geneartion df {generation_df.shape}")
    print(f"Production df {production_df.shape}")

    # List to collect summary data
    summary_data = []

    for time in range(T):
        production_t = production_df[production_df['time_step'] == (time + 1)]
        loss_of_load_t = loss_of_load_df[loss_of_load_df['time_step'] == (time + 1)]

        if not loss_of_load_t.empty:
            loss_load = loss_of_load_t['loss_of_load'].sum()
            loss_cost = loss_load * value_of_lost_load
        else:
            loss_cost = 0

        unique_pairs = production_t[['location', 'technology']].drop_duplicates().values
        total_prod_cost = 0

        for location, tech in unique_pairs:
            generation_cost = generation_df[
                (generation_df['location'] == location) & 
                (generation_df['technology'] == tech)
            ]['variable_cost'].values[0]

            production_rows = production_t[
                (production_t['location'] == location) & 
                (production_t['technology'] == tech)
            ]

     

            total_production = production_rows['production'].sum()
            cost = total_production * generation_cost
            total_prod_cost += cost

        # Add to summary
        summary_data.append({
            'index': time + 1,
            'operational_cost': total_prod_cost,
            'loss_cost': loss_cost,
            'objective_value': total_prod_cost+loss_cost,
            'inference_time': individual_runtime
        })

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    # # Save to CSV
    summary_df.to_csv(f"{base_path}/summary.csv", index=False)
    print(f"Summary saved to {base_path}/summary.csv")
    
if __name__ == "__main__":
    base_path = "Test_instances_input/2Nodes-no-ren"
    generate_summary_data_solver(base_path)
    print("Summary data generated successfully.")