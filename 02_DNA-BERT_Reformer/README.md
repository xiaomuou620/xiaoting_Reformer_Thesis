# input data files
/data/0713_full_combined_data.h5: final data used in training, which is too large to put on github

# training
train_reformer_bc.sh: run it ``./train_reformer_bc.sh``, settings include python file: train_reformer_bc_722.py, which will output models and training and validation results to /M0_722_e10

# test
00_performance_test.sh: run it ``./00_performance_test.sh", it will run python file 00_performance_test_726.py, which will output test performance
