import pickle

# basic
train_process = False
random_state = 0
replace_zero_to_nan = False
generate_polynomial_features = False
reduce_dimension = False



# drop_na, imputation, linear_regression, bayesian_ridge, decision_tree_regressor,
# extra_tree_regressor, k_neighbors_regressor
deal_with_missing_value = 'k_neighbors_regressor'


# box-cox, yeo-johnson
variable_transformation = 'yeo-johnson'
skew_threshold = 0.05

# create group feature
kmeans_standardization = 'min_max'
kmeans_k_range = range(2,20)



# Feature Selection
with open(f'../Data/meta/train_x_after_feature_engineering.pickle', 'rb') as f:
    train_x_after_fe = pickle.load(f)
feature_select = train_x_after_fe.columns




# standardization
# min_max_scaler, standard_scaler
standardization = 'standard_scaler'
