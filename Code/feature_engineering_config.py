# basic
train_process = True
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

# outlier detection
# ecod, suod, elliptic_envelope
outlier_detection = 'suod'

ecod_contamination = 0.03
ecod_threshold = 20

suod_threshold = 2


# create group feature
kmeans_standardization = 'min_max'
kmeans_k_range = range(2,20)

# generate polynomial features
pf_degree = 2
pf_interaction_only = False


# feature selection
feature_selection = ['variance', 'recursive_feature_elimination']
min_features_to_select = 50
feature_selection_cv = 3


# standardization
# min_max_scaler, standard_scaler
standardization = 'standard_scaler'
