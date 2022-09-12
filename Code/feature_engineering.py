# Import Libraries
import pandas as pd
import numpy as np
import pickle
# imputattion
from sklearn.impute import SimpleImputer
## To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
# Variable transformation
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# outlier detection
from pyod.models.ecod import ECOD
from pyod.models.suod import SUOD
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from sklearn.covariance import EllipticEnvelope
# Drop feature
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
# Create group feature
from utils import kmeans_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Generate polynomial and interaction features
from sklearn.preprocessing import PolynomialFeatures
# Reduce dimension
from sklearn.decomposition import PCA
# Standardization
from sklearn import preprocessing



class DealWithMissingValue:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
    
    def drop_na(self) -> pd.DataFrame:
        return self.dataframe.dropna()
    
    def imputation(self, strategy: str, train: bool, fill_value=0) -> pd.DataFrame:
        '''
        strategy: most_frequent, median, mean, constant
        '''
        if train:
            imputer_most_frequent = SimpleImputer(
                missing_values=np.nan, strategy=strategy, fill_value=fill_value
            )
            imputer_most_frequent.fit(self.dataframe)
            with open('../Model/imputer_most_frequent.pickle', 'wb' ) as f:
                pickle.dump(imputer_most_frequent, f)
        else:
            with open('../Model/imputer_most_frequent.pickle', 'rb') as f:
                imputer_most_frequent = pickle.load(f)
                
        res = pd.DataFrame(
            imputer_most_frequent.transform(self.dataframe),  # Remake DataFrame with new values
            index=self.dataframe.index,
            columns=self.dataframe.columns
        )

        return res
        
    def iterative_imputer(self, method: str, train: bool) -> pd.DataFrame:
        model_dic = {
            'linear_regression': LinearRegression(),
            'bayesian_ridge': BayesianRidge(),
            'decision_tree_regressor': DecisionTreeRegressor(max_features='sqrt', random_state=0),
            'extra_tree_regressor': ExtraTreesRegressor(n_estimators=10, random_state=0),
            'k_neighbors_regressor': KNeighborsRegressor(n_neighbors=5)
        }
        if train:
            iterative_imputer = IterativeImputer(random_state=0, estimator=model_dic[method])        
            iterative_imputer.fit(self.dataframe)
            with open(f'../Model/imputer_by_{method}.pickle', 'wb' ) as f:
                pickle.dump(iterative_imputer, f)
        else:
            with open(f'../Model/imputer_by_{method}.pickle', 'rb') as f:
                iterative_imputer = pickle.load(f)
        
        res = pd.DataFrame(
            iterative_imputer.transform(self.dataframe),  # Remake DataFrame with new values
            index=self.dataframe.index,
            columns=self.dataframe.columns
        )
        
        return res
    
    
class VariableTransformation:
    def __init__(self, dataframe: pd.DataFrame, col: str) -> None:
        self.dataframe = dataframe
        self.col = col
    def transform(self, method, train: bool) -> pd.DataFrame:
        if train:
            pt = PowerTransformer(method=method)  # box-cox, yeo-johnson
            pt.fit( np.array(self.dataframe[self.col]).reshape(-1,1) )  # 會出現 RuntimeWarning，原因是小數點溢出
            with open(f'../Model/variable_transformation_{method}_{self.col}.pickle', 'wb' ) as f:
                pickle.dump(pt, f)
        else:
            with open(f'../Model/variable_transformation_{method}_{self.col}.pickle', 'rb') as f:
                pt = pickle.load(f)
        
        res = pt.transform( np.array(self.dataframe[self.col]).reshape(-1,1) )
        res = res.reshape(res.shape[0])
        return res

    
class OutlierDetection:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
    
    def ecod(self, contamination, threshold, train: bool) -> np.ndarray:
        if train:
            clf = ECOD(contamination=contamination)  # 0.03
            # The contamination indicates the expected percentage of outliers in the training data.
            clf.fit(self.dataframe)
            with open(f'../Model/outlier_detection_ecod.pickle', 'wb' ) as f:
                pickle.dump(clf, f)
        else:
            with open(f'../Model/outlier_detection_ecod.pickle', 'rb') as f:
                clf = pickle.load(f)
        
        return clf.decision_function(self.dataframe) >= threshold
    
    def suod(self, detector_list: list, threshold: int, train: bool, verbose: bool, combination='average',  n_jobs=-1) -> np.ndarray:
        if train:
            # decide the number of parallel process, and the combination method
            # then clf can be used as any outlier detection model
            clf = SUOD(
                base_estimators=detector_list,
                n_jobs=n_jobs,
                combination=combination,
                verbose=verbose
            )
            clf.fit(self.dataframe)  # fit all models with X
            with open(f'../Model/outlier_detection_suod.pickle', 'wb' ) as f:
                pickle.dump(clf, f)
        else:
            with open(f'../Model/outlier_detection_suod.pickle', 'rb') as f:
                clf = pickle.load(f)
                
        return clf.decision_function(self.dataframe) >= threshold
        
    def elliptic_envelope(self, random_state, train) -> np.ndarray:
        if train:
            cov = EllipticEnvelope(random_state=random_state).fit(self.dataframe)
            with open(f'../Model/outlier_detection_elliptic_envelope.pickle', 'wb' ) as f:
                pickle.dump(cov, f)
        else:
            with open(f'../Model/outlier_detection_elliptic_envelope.pickle', 'rb') as f:
                cov = pickle.load(f)
        
        # predict returns 1 for an inlier and -1 for an outlier
        return cov.predict(self.dataframe) == -1
        
        
class FeatureSelection:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        
    def variance(self, threshold, train) -> pd.DataFrame:
        if train:
            vt_selector = VarianceThreshold(threshold=threshold)  # variance = 0 的 column 會被移除
            vt_selector.fit(self.dataframe._get_numeric_data())
            with open(f'../Model/drop_feature_by_variance.pickle', 'wb' ) as f:
                pickle.dump(vt_selector, f)
        else:
            with open(f'../Model/drop_feature_by_variance.pickle', 'rb') as f:
                vt_selector = pickle.load(f)
        
        numeric_data = vt_selector.transform(self.dataframe._get_numeric_data())
        dataframe_copy = self.dataframe.copy()
        dataframe_copy[dataframe_copy._get_numeric_data().columns] = numeric_data
        return dataframe_copy
        # return self.dataframe.loc[:, ~vt_selector.get_support()], self.dataframe.loc[:, vt_selector.get_support()]
    
    
    def mutual_information(self, k, train, y=None) -> pd.DataFrame:
        if train:
            select_best_k_feature = SelectKBest(
                mutual_info_regression, k=k
            ).fit(self.dataframe, y)
            with open(f'../Model/drop_feature_by_mutual_info.pickle', 'wb' ) as f:
                pickle.dump(select_best_k_feature, f)
        else:
            with open(f'../Model/drop_feature_by_mutual_info.pickle', 'rb') as f:
                select_best_k_feature = pickle.load(f)
        
        return self.dataframe.loc[:, select_best_k_feature.get_support()]
    
    
    def importance_weight(self, y, estimator, train) -> pd.DataFrame:
        '''
        Meta-transformer for selecting features based on importance weights.
        '''
        if train:
            estimator.fit(self.dataframe, y)
            model = SelectFromModel(estimator, prefit=True)
    
            with open(f'../Model/drop_feature_by_importance_weight.pickle', 'wb' ) as f:
                pickle.dump(model, f)
        else:
            with open(f'../Model/drop_feature_by_importance_weight.pickle', 'rb') as f:
                model = pickle.load(f)
        
        return self.dataframe.loc[:, model.get_support()]
    
    def recursive_feature_elimination(self, y, estimator, cv, min_features_to_select, train, n_jobs=-1) -> pd.DataFrame:
        if train:
            rfecv_feature_selector = RFECV(
                estimator,
                cv=cv,
                n_jobs=n_jobs,
                scoring='neg_mean_squared_error',
                min_features_to_select=min_features_to_select
            )
            rfecv_feature_selector = rfecv_feature_selector.fit(self.dataframe, y)
            with open(f'../Model/drop_feature_by_recursive_feature_elimination_with_cv.pickle', 'wb' ) as f:
                pickle.dump(rfecv_feature_selector, f)
        else:
            with open(f'../Model/drop_feature_by_recursive_feature_elimination_with_cv.pickle', 'rb') as f:
                rfecv_feature_selector = pickle.load(f)
        
        return self.dataframe.loc[:, rfecv_feature_selector.support_]
    
    
    
    
    
class CreateGroupFeatureFromAllCol:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
    
    def kmeans_with_auto_k(self, standardization: str, k_range: range, random_state: int, parallel: bool, parallel_verbose: int, train: bool) -> np.ndarray:
        standardization_dic = {
            'min_max': MinMaxScaler(),
            'zscore': StandardScaler()
        }
        
        if train:
            # scale the data
            scaler = standardization_dic[standardization]
            scaler.fit(self.dataframe)
            with open(f'../Model/create_group_feature_from_all_column_by_kmeans_with_{standardization}.pickle', 'wb' ) as f:
                pickle.dump(scaler, f)
                
            scaled_data = scaler.transform(self.dataframe)
            best_k, results = kmeans_utils.choose_best_k_for_kmeans(
                    scaled_data,
                    k_range=k_range,
                    verbose = parallel_verbose,
                    parallel=parallel
                )         
            kmeans = KMeans(n_clusters=best_k, random_state=random_state).fit(scaled_data)
            with open(f'../Model/create_group_feature_from_all_column_by_kmeans.pickle', 'wb' ) as f:
                pickle.dump(kmeans, f)

        else:
            with open(f'../Model/create_group_feature_from_all_column_by_kmeans_with_{standardization}.pickle', 'rb') as f:
                scaler = pickle.load(f)
            
            with open(f'../Model/create_group_feature_from_all_column_by_kmeans.pickle', 'rb') as f:
                kmeans = pickle.load(f)
            
        scaled_data = scaler.transform(self.dataframe)
        return kmeans.predict(scaled_data)
    
    
class CreateGroupFeatureFromEachCol:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
    
    def kmeans_with_auto_k(self, standardization: str, k_range: range, random_state: int, parallel: bool, parallel_verbose: int, train: bool) -> np.ndarray:
        standardization_dic = {
            'min_max': MinMaxScaler(),
            'zscore': StandardScaler()
        }
        
        if train:
            # scale the data
            scaler = standardization_dic[standardization]
            scaler.fit(self.dataframe)
            with open(f'../Model/create_group_feature_from_each_column_by_kmeans_with_{standardization}.pickle', 'wb' ) as f:
                pickle.dump(scaler, f)
                
            scaled_data = scaler.transform(self.dataframe)
            best_k, results = kmeans_utils.choose_best_k_for_kmeans(
                    scaled_data,
                    k_range=k_range,
                    verbose = parallel_verbose,
                    parallel=parallel
                )         
            kmeans = KMeans(n_clusters=best_k, random_state=random_state).fit(scaled_data)
            with open(f'../Model/create_group_feature_from_each_column_by_kmeans.pickle', 'wb' ) as f:
                pickle.dump(kmeans, f)

        else:
            with open(f'../Model/create_group_feature_from_each_column_by_kmeans_with_{standardization}.pickle', 'rb') as f:
                scaler = pickle.load(f)
            
            with open(f'../Model/create_group_feature_from_each_column_by_kmeans.pickle', 'rb') as f:
                kmeans = pickle.load(f)
            
        scaled_data = scaler.transform(self.dataframe)
        return kmeans.predict(scaled_data)    
    
    
    
    
class GeneratePolynomialFeatures:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        
    def get_dataframe(self, degree, interaction_only, train) -> pd.DataFrame:
        if train:
            pf = PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=False
            )
            pf.fit(self.dataframe)
            with open(f'../Model/polynomial_features.pickle', 'wb' ) as f:
                pickle.dump(pf, f)
            
        else:
            with open(f'../Model/polynomial_features.pickle', 'rb') as f:
                pf = pickle.load(f)
            

        return pd.DataFrame(
            pf.transform(self.dataframe),
            columns=pf.get_feature_names(self.dataframe.columns)
        )
    
    
class ReduceDimensionPCA:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
    
    def pca(self, n_components, train) -> np.ndarray:
        if train:
            pca_model = PCA(n_components=n_components)
            pca_model.fit(self.dataframe)
            
            with open(f'../Model/pca_model.pickle', 'wb' ) as f:
                pickle.dump(pca_model, f)
            
        else:
            with open(f'../Model/pca_model.pickle', 'rb') as f:
                pca_model = pickle.load(f)
        
        return pca_model.transform(self.dataframe)



class Standardization:
    def __init__(self, dataframe: pd.DataFrame, prefix: str) -> None:
        self.dataframe = dataframe
        self.prefix = prefix
        
    def standard_scaler(self, train) -> pd.DataFrame:
        if train:
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.dataframe._get_numeric_data().values) # returns a numpy array first
            with open(f'../Model/standard_scaler_{self.prefix}.pickle', 'wb' ) as f:
                pickle.dump(scaler, f)
        else:
            with open(f'../Model/standard_scaler_{self.prefix}.pickle', 'rb') as f:
                scaler = pickle.load(f)
                
        data_scaled = scaler.transform(self.dataframe._get_numeric_data().values)
        dataframe_copy = self.dataframe.copy()
        dataframe_copy[dataframe_copy._get_numeric_data().columns] = data_scaled
        return dataframe_copy
    
        
    def min_max_scaler(self, train) -> pd.DataFrame:
        if train:
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(self.dataframe.values) # returns a numpy array first
            with open(f'../Model/min_max_scaler.pickle', 'wb' ) as f:
                pickle.dump(scaler, f)
        else:
            with open(f'../Model/min_max_scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)
                
        data_scaled = scaler.transform(self.dataframe._get_numeric_data().values)
        dataframe_copy = self.dataframe.copy()
        dataframe_copy[dataframe_copy._get_numeric_data().columns] = data_scaled
        return dataframe_copy
