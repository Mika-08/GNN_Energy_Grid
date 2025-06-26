using GenerationExpansionPlanning
using Gurobi

# Step 1: Read the experiment config
@info "Reading the config"
config_path = "config.toml"
# config_path = "config.toml"
config = read_config(config_path)

# Step 2: Parse the data
@info "Parsing the config data"
experiment_data = ExperimentData(config[:input])

# Step 3: Run the experiments
@info "Running the experiments defined by $config_path"
elapsed_time = @elapsed begin
    experiment_result = run_experiment(experiment_data, Gurobi.Optimizer)
end
@info "Experiment completed in $(round(elapsed_time, digits=2)) seconds"
# @info "Gurobi solver time: $(round(experiment_result.solver_time_seconds[1], digits=2)) seconds"
println("Solver times:")
for (i, t) in enumerate(experiment_result.solver_time_seconds)
    # println("  Run $i: $(t) seconds")
    @info "- Run $i: $(round(t, digits=2)) seconds"
end
# Step 4: Save the results
output_config = config[:output]
save_result(experiment_result, output_config)
