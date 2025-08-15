
# input data files
/data/0713_full_combined_data.h5: final data used in training

# training
train_reformer_bc_plants_M*.sh: run it ``./train_reformer_bc_plants_M*.sh``, settings include python file: train_reformer_bc__plants_722.py, which will output models and training and validation results to /M*_722

# test
00_performance_test_plant_M*.sh: run it ``./00_performance_test_plant_M*.sh", it will run python file 00_performance_test_plant_722.py, which will output test performance to /M*_722
