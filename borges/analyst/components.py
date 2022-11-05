"""
analyst.components:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from types import ModuleType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import borges
from . import base


@dataclasses.dataclass
class AnalystProcess(amos.project.Process):
    """Base class for parts of a data science project workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout amos. For example, if a 
            amos instance needs settings from an Idea instance, 
            'name' should match the appropriate section name in an Idea 
            instance. Defaults to None. 
        contents (Any): stored item(s) for use by a Process subclass instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parameters (Mapping[Any, Any]]): parameters to be attached to 'contents' 
            when the 'implement' method is called. Defaults to an empty dict.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be at the end of a parallel workflow structure. Defaults to 
            False.
        after_split (ClassVar[bool]): whether the instance's method should
            only be called after the data is split into training and testing
            sets. Defaults to False.
        before_split (ClassVar[bool]): whether the instance's method should
            only be called before the data is split into training and testing
            sets. Defaults to False.
        model_limts (ClassVar[bool]): any model types that the method must be
            used with. If None are listed, borges assumes that the instance is
            compatible with all model types. Defaults to an empty list.
                
    """
    name: str = None
    contents: Any = None
    iterations: Union[int, str] = 1
    parameters: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    parallel: ClassVar[bool] = False
    after_split: ClassVar[bool] = False
    before_split: ClassVar[bool] = False
    model_limits: ClassVar[Sequence[str]] = []
    
    """ Public Methods """
    
    def implement(self, data: borges.core.Dataset, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
            
        """
        if data.split and self.before_split:
            raise ValueError(
                f'{self.name} component can only be used with unsplit data')
        elif not data.split and self.after_split:
            raise ValueError(
                f'{self.name} component can only be used with split data')            
        if self.parameters:
            parameters = self.parameters
            parameters.update(kwargs)
        else:
            parameters = kwargs
        if self.contents not in [None, 'None', 'none']:
            data = self.contents.implement(data = data, **parameters)
        return data

@dataclasses.dataclass
class TheoryFill(base.TheoryStep):
    
    name: str = 'fill'
    parameters: Dict[str, Any] = dataclasses.field(default_factory = lambda: {
        'boolean': False,
        'float': 0.0,
        'integer': 0,
        'string': '',
        'categorical': '',
        'list': [],
        'datetime': 1/1/1900,
        'timedelta': 0})


@dataclasses.dataclass
class TheoryCategorize(base.TheoryStep):
    
    name: str = 'categorize'


@dataclasses.dataclass
class TheoryScale(base.TheoryStep):
    
    name: str = 'scale'


@dataclasses.dataclass
class TheorySplit(base.TheoryStep):
    
    name: str = 'split'


@dataclasses.dataclass
class TheoryEncode(base.TheoryStep):
    
    name: str = 'encode'


@dataclasses.dataclass
class TheoryMix(base.TheoryStep):
    
    name: str = 'mix'


@dataclasses.dataclass
class TheoryCleave(base.TheoryStep):
    
    name: str = 'cleave'


@dataclasses.dataclass
class TheorySample(base.TheoryStep):
    
    name: str = 'sample'


@dataclasses.dataclass
class TheoryReduce(base.TheoryStep):
    
    name: str = 'reduce'


@dataclasses.dataclass
class TheoryModel(base.TheoryStep):
    
    name: str = 'model'


@dataclasses.dataclass
class TheoryKNNImputer(base.TheoryTechnique):
    
    name: str = 'knn_imputer'
    module: str = 'self'
    contents: str = 'knn_impute'


@dataclasses.dataclass
class TheoryAutomaticCategorizor(base.TheoryTechnique):
    
    name: str = 'automatic_categorizer'
    module: str = 'self'
    contents: str = 'auto_categorize'


@dataclasses.dataclass
class TheoryMaxAbs(base.TheoryTechnique):
    
    name: str = 'maximum_absolute_value_scaler'
    module: str = 'sklearn.preprocessing'
    contents: str = 'MaxAbsScaler'
    default: Dict[str, Any] = dataclasses.field(default_factory = lambda: 
        {'copy': False})


@dataclasses.dataclass
class TheoryKfold(base.TheoryTechnique):
    
    name: str = 'Kfold_splitter'
    module: str = 'sklearn.model_selection'
    contents: str = 'KFold'
    default: Dict[str, Any] = dataclasses.field(default_factory = lambda: 
        {'n_splits': 5, 
         'shuffle': False})


@dataclasses.dataclass
class TheoryKfold(base.TheoryTechnique):
    
    name: str = 'Kfold_splitter'
    module: str = 'sklearn.model_selection'
    contents: str = 'KFold'
    default: Dict[str, Any] = dataclasses.field(default_factory = lambda: 
        {'n_splits': 5, 
         'shuffle': False})


@dataclasses.dataclass
class TheoryXGBoost(base.TheoryTechnique):

    name: str = 'xgboost'
    module: str = 'xgboost'
    contents: str = 'XGBClassifier'
       

# raw_options: Dict[str, borges.TheoryTechnique] = {
#     'fill': {
#         'defaults': borges.TheoryTechnique(
#             name = 'defaults',
#             module = 'borges.analyst.algorithms',
#             algorithm = 'smart_fill',
#             default = {'defaults': {
#                 'boolean': False,
#                 'float': 0.0,
#                 'integer': 0,
#                 'string': '',
#                 'categorical': '',
#                 'list': [],
#                 'datetime': 1/1/1900,
#                 'timedelta': 0}}),
#         'impute': borges.TheoryTechnique(
#             name = 'defaults',
#             module = 'sklearn.impute',
#             algorithm = 'TheoryImputer',
#             default = {'defaults': {}}),
#         'knn_impute': borges.TheoryTechnique(
#             name = 'defaults',
#             module = 'sklearn.impute',
#             algorithm = 'KNNImputer',
#             default = {'defaults': {}})},
#     'categorize': {
#         'automatic': borges.TheoryTechnique(
#             name = 'automatic',
#             module = 'borges.analyst.algorithms',
#             algorithm = 'auto_categorize',
#             default = {'threshold': 10}),
#         'binary': borges.TheoryTechnique(
#             name = 'binary',
#             module = 'sklearn.preprocessing',
#             algorithm = 'Binarizer',
#             default = {'threshold': 0.5}),
#         'bins': borges.TheoryTechnique(
#             name = 'bins',
#             module = 'sklearn.preprocessing',
#             algorithm = 'KBinsDiscretizer',
#             default = {
#                 'strategy': 'uniform',
#                 'n_bins': 5},
#             selected = True,
#             required = {'encode': 'onehot'})},
#     'scale': {
#         'gauss': borges.TheoryTechnique(
#             name = 'gauss',
#             module = None,
#             algorithm = 'Gaussify',
#             default = {'standardize': False, 'copy': False},
#             selected = True,
#             required = {'rescaler': 'standard'}),
#         'maxabs': borges.TheoryTechnique(
#             name = 'maxabs',
#             module = 'sklearn.preprocessing',
#             algorithm = 'MaxAbsScaler',
#             default = {'copy': False},
#             selected = True),
#         'minmax': borges.TheoryTechnique(
#             name = 'minmax',
#             module = 'sklearn.preprocessing',
#             algorithm = 'MinMaxScaler',
#             default = {'copy': False},
#             selected = True),
#         'normalize': borges.TheoryTechnique(
#             name = 'normalize',
#             module = 'sklearn.preprocessing',
#             algorithm = 'Normalizer',
#             default = {'copy': False},
#             selected = True),
#         'quantile': borges.TheoryTechnique(
#             name = 'quantile',
#             module = 'sklearn.preprocessing',
#             algorithm = 'QuantileTransformer',
#             default = {'copy': False},
#             selected = True),
#         'robust': borges.TheoryTechnique(
#             name = 'robust',
#             module = 'sklearn.preprocessing',
#             algorithm = 'RobustScaler',
#             default = {'copy': False},
#             selected = True),
#         'standard': borges.TheoryTechnique(
#             name = 'standard',
#             module = 'sklearn.preprocessing',
#             algorithm = 'StandardScaler',
#             default = {'copy': False},
#             selected = True)},
#     'split': {
#         'group_kfold': borges.TheoryTechnique(
#             name = 'group_kfold',
#             module = 'sklearn.model_selection',
#             algorithm = 'GroupKFold',
#             default = {'n_splits': 5},
#             runtime = {'random_state': 'seed'},
#             selected = True,
#             fit_method = None,
#             transform_method = 'split'),
#         'kfold': borges.TheoryTechnique(
#             name = 'kfold',
#             module = 'sklearn.model_selection',
#             algorithm = 'KFold',
#             default = {'n_splits': 5, 'shuffle': False},
#             runtime = {'random_state': 'seed'},
#             selected = True,
#             required = {'shuffle': True},
#             fit_method = None,
#             transform_method = 'split'),
#         'stratified': borges.TheoryTechnique(
#             name = 'stratified',
#             module = 'sklearn.model_selection',
#             algorithm = 'StratifiedKFold',
#             default = {'n_splits': 5, 'shuffle': False},
#             runtime = {'random_state': 'seed'},
#             selected = True,
#             required = {'shuffle': True},
#             fit_method = None,
#             transform_method = 'split'),
#         'time': borges.TheoryTechnique(
#             name = 'time',
#             module = 'sklearn.model_selection',
#             algorithm = 'TimeSeriesSplit',
#             default = {'n_splits': 5},
#             runtime = {'random_state': 'seed'},
#             selected = True,
#             fit_method = None,
#             transform_method = 'split'),
#         'train_test': borges.TheoryTechnique(
#             name = 'train_test',
#             module = 'sklearn.model_selection',
#             algorithm = 'ShuffleSplit',
#             default = {'test_size': 0.33},
#             runtime = {'random_state': 'seed'},
#             required = {'n_splits': 1},
#             selected = True,
#             fit_method = None,
#             transform_method = 'split')},
#     'encode': {
#         'backward': borges.TheoryTechnique(
#             name = 'backward',
#             module = 'category_encoders',
#             algorithm = 'BackwardDifferenceEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'basen': borges.TheoryTechnique(
#             name = 'basen',
#             module = 'category_encoders',
#             algorithm = 'BaseNEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'binary': borges.TheoryTechnique(
#             name = 'binary',
#             module = 'category_encoders',
#             algorithm = 'BinaryEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'dummy': borges.TheoryTechnique(
#             name = 'dummy',
#             module = 'category_encoders',
#             algorithm = 'OneHotEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'hashing': borges.TheoryTechnique(
#             name = 'hashing',
#             module = 'category_encoders',
#             algorithm = 'HashingEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'helmert': borges.TheoryTechnique(
#             name = 'helmert',
#             module = 'category_encoders',
#             algorithm = 'HelmertEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'james_stein': borges.TheoryTechnique(
#             name = 'james_stein',
#             module = 'category_encoders',
#             algorithm = 'JamesSteinEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'loo': borges.TheoryTechnique(
#             name = 'loo',
#             module = 'category_encoders',
#             algorithm = 'LeaveOneOutEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'm_estimate': borges.TheoryTechnique(
#             name = 'm_estimate',
#             module = 'category_encoders',
#             algorithm = 'MEstimateEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'ordinal': borges.TheoryTechnique(
#             name = 'ordinal',
#             module = 'category_encoders',
#             algorithm = 'OrdinalEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'polynomial': borges.TheoryTechnique(
#             name = 'polynomial_encoder',
#             module = 'category_encoders',
#             algorithm = 'PolynomialEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'sum': borges.TheoryTechnique(
#             name = 'sum',
#             module = 'category_encoders',
#             algorithm = 'SumEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'target': borges.TheoryTechnique(
#             name = 'target',
#             module = 'category_encoders',
#             algorithm = 'TargetEncoder',
#             data_dependent = {'cols': 'categoricals'}),
#         'woe': borges.TheoryTechnique(
#             name = 'weight_of_evidence',
#             module = 'category_encoders',
#             algorithm = 'WOEEncoder',
#             data_dependent = {'cols': 'categoricals'})},
#     'mix': {
#         'polynomial': borges.TheoryTechnique(
#             name = 'polynomial_mixer',
#             module = 'sklearn.preprocessing',
#             algorithm = 'PolynomialFeatures',
#             default = {
#                 'degree': 2,
#                 'interaction_only': True,
#                 'include_bias': True}),
#         'quotient': borges.TheoryTechnique(
#             name = 'quotient',
#             module = None,
#             algorithm = 'QuotientFeatures'),
#         'sum': borges.TheoryTechnique(
#             name = 'sum',
#             module = None,
#             algorithm = 'SumFeatures'),
#         'difference': borges.TheoryTechnique(
#             name = 'difference',
#             module = None,
#             algorithm = 'DifferenceFeatures')},
#     'cleave': {
#         'cleaver': borges.TheoryTechnique(
#             name = 'cleaver',
#             module = 'borges.analyst.algorithms',
#             algorithm = 'Cleaver')},
#     'sample': {
#         'adasyn': borges.TheoryTechnique(
#             name = 'adasyn',
#             module = 'imblearn.over_sampling',
#             algorithm = 'ADASYN',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'cluster': borges.TheoryTechnique(
#             name = 'cluster',
#             module = 'imblearn.under_sampling',
#             algorithm = 'ClusterCentroids',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'knn': borges.TheoryTechnique(
#             name = 'knn',
#             module = 'imblearn.under_sampling',
#             algorithm = 'AllKNN',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'near_miss': borges.TheoryTechnique(
#             name = 'near_miss',
#             module = 'imblearn.under_sampling',
#             algorithm = 'NearMiss',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'random_over': borges.TheoryTechnique(
#             name = 'random_over',
#             module = 'imblearn.over_sampling',
#             algorithm = 'RandomOverSampler',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'random_under': borges.TheoryTechnique(
#             name = 'random_under',
#             module = 'imblearn.under_sampling',
#             algorithm = 'RandomUnderSampler',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'smote': borges.TheoryTechnique(
#             name = 'smote',
#             module = 'imblearn.over_sampling',
#             algorithm = 'SMOTE',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'smotenc': borges.TheoryTechnique(
#             name = 'smotenc',
#             module = 'imblearn.over_sampling',
#             algorithm = 'SMOTENC',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             data_dependent = {
#                 'categorical_features': 'categoricals_indices'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'smoteenn': borges.TheoryTechnique(
#             name = 'smoteenn',
#             module = 'imblearn.combine',
#             algorithm = 'SMOTEENN',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample'),
#         'smotetomek': borges.TheoryTechnique(
#             name = 'smotetomek',
#             module = 'imblearn.combine',
#             algorithm = 'SMOTETomek',
#             default = {'sampling_strategy': 'auto'},
#             runtime = {'random_state': 'seed'},
#             fit_method = None,
#             transform_method = 'fit_resample')},
#     'reduce': {
#         'kbest': borges.TheoryTechnique(
#             name = 'kbest',
#             module = 'sklearn.feature_selection',
#             algorithm = 'SelectKBest',
#             default = {'k': 10, 'score_func': 'f_classif'},
#             selected = True),
#         'fdr': borges.TheoryTechnique(
#             name = 'fdr',
#             module = 'sklearn.feature_selection',
#             algorithm = 'SelectFdr',
#             default = {'alpha': 0.05, 'score_func': 'f_classif'},
#             selected = True),
#         'fpr': borges.TheoryTechnique(
#             name = 'fpr',
#             module = 'sklearn.feature_selection',
#             algorithm = 'SelectFpr',
#             default = {'alpha': 0.05, 'score_func': 'f_classif'},
#             selected = True),
#         'custom': borges.TheoryTechnique(
#             name = 'custom',
#             module = 'sklearn.feature_selection',
#             algorithm = 'SelectFromModel',
#             default = {'threshold': 'mean'},
#             runtime = {'estimator': 'algorithm'},
#             selected = True),
#         'rank': borges.TheoryTechnique(
#             name = 'rank',
#             module = 'borges.critic.rank',
#             algorithm = 'RankSelect',
#             selected = True),
#         'rfe': borges.TheoryTechnique(
#             name = 'rfe',
#             module = 'sklearn.feature_selection',
#             algorithm = 'RFE',
#             default = {'n_features_to_select': 10, 'step': 1},
#             runtime = {'estimator': 'algorithm'},
#             selected = True),
#         'rfecv': borges.TheoryTechnique(
#             name = 'rfecv',
#             module = 'sklearn.feature_selection',
#             algorithm = 'RFECV',
#             default = {'n_features_to_select': 10, 'step': 1},
#             runtime = {'estimator': 'algorithm'},
#             selected = True)}}

# raw_model_options: Dict[str, borges.TheoryTechnique] = {
#     'classify': {
#         'adaboost': borges.TheoryTechnique(
#             name = 'adaboost',
#             module = 'sklearn.ensemble',
#             algorithm = 'AdaBoostClassifier',
#             transform_method = None),
#         'baseline_classifier': borges.TheoryTechnique(
#             name = 'baseline_classifier',
#             module = 'sklearn.dummy',
#             algorithm = 'DummyClassifier',
#             required = {'strategy': 'most_frequent'},
#             transform_method = None),
#         'logit': borges.TheoryTechnique(
#             name = 'logit',
#             module = 'sklearn.linear_model',
#             algorithm = 'LogisticRegression',
#             transform_method = None),
#         'random_forest': borges.TheoryTechnique(
#             name = 'random_forest',
#             module = 'sklearn.ensemble',
#             algorithm = 'RandomForestClassifier',
#             transform_method = None),
#         'svm_linear': borges.TheoryTechnique(
#             name = 'svm_linear',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'linear', 'probability': True},
#             transform_method = None),
#         'svm_poly': borges.TheoryTechnique(
#             name = 'svm_poly',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'poly', 'probability': True},
#             transform_method = None),
#         'svm_rbf': borges.TheoryTechnique(
#             name = 'svm_rbf',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'rbf', 'probability': True},
#             transform_method = None),
#         'svm_sigmoid': borges.TheoryTechnique(
#             name = 'svm_sigmoid ',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'sigmoid', 'probability': True},
#             transform_method = None),
#         'tensorflow': borges.TheoryTechnique(
#             name = 'tensorflow',
#             module = 'tensorflow',
#             algorithm = None,
#             default = {
#                 'batch_size': 10,
#                 'epochs': 2},
#             transform_method = None),
#         'xgboost': borges.TheoryTechnique(
#             name = 'xgboost',
#             module = 'xgboost',
#             algorithm = 'XGBClassifier',
#             # data_dependent = 'scale_pos_weight',
#             transform_method = None)},
#     'cluster': {
#         'affinity': borges.TheoryTechnique(
#             name = 'affinity',
#             module = 'sklearn.cluster',
#             algorithm = 'AffinityPropagation',
#             transform_method = None),
#         'agglomerative': borges.TheoryTechnique(
#             name = 'agglomerative',
#             module = 'sklearn.cluster',
#             algorithm = 'AgglomerativeClustering',
#             transform_method = None),
#         'birch': borges.TheoryTechnique(
#             name = 'birch',
#             module = 'sklearn.cluster',
#             algorithm = 'Birch',
#             transform_method = None),
#         'dbscan': borges.TheoryTechnique(
#             name = 'dbscan',
#             module = 'sklearn.cluster',
#             algorithm = 'DBSCAN',
#             transform_method = None),
#         'kmeans': borges.TheoryTechnique(
#             name = 'kmeans',
#             module = 'sklearn.cluster',
#             algorithm = 'KMeans',
#             transform_method = None),
#         'mean_shift': borges.TheoryTechnique(
#             name = 'mean_shift',
#             module = 'sklearn.cluster',
#             algorithm = 'MeanShift',
#             transform_method = None),
#         'spectral': borges.TheoryTechnique(
#             name = 'spectral',
#             module = 'sklearn.cluster',
#             algorithm = 'SpectralClustering',
#             transform_method = None),
#         'svm_linear': borges.TheoryTechnique(
#             name = 'svm_linear',
#             module = 'sklearn.cluster',
#             algorithm = 'OneClassSVM',
#             transform_method = None),
#         'svm_poly': borges.TheoryTechnique(
#             name = 'svm_poly',
#             module = 'sklearn.cluster',
#             algorithm = 'OneClassSVM',
#             transform_method = None),
#         'svm_rbf': borges.TheoryTechnique(
#             name = 'svm_rbf',
#             module = 'sklearn.cluster',
#             algorithm = 'OneClassSVM,',
#             transform_method = None),
#         'svm_sigmoid': borges.TheoryTechnique(
#             name = 'svm_sigmoid',
#             module = 'sklearn.cluster',
#             algorithm = 'OneClassSVM',
#             transform_method = None)},
#     'regress': {
#         'adaboost': borges.TheoryTechnique(
#             name = 'adaboost',
#             module = 'sklearn.ensemble',
#             algorithm = 'AdaBoostRegressor',
#             transform_method = None),
#         'baseline_regressor': borges.TheoryTechnique(
#             name = 'baseline_regressor',
#             module = 'sklearn.dummy',
#             algorithm = 'DummyRegressor',
#             required = {'strategy': 'mean'},
#             transform_method = None),
#         'bayes_ridge': borges.TheoryTechnique(
#             name = 'bayes_ridge',
#             module = 'sklearn.linear_model',
#             algorithm = 'BayesianRidge',
#             transform_method = None),
#         'lasso': borges.TheoryTechnique(
#             name = 'lasso',
#             module = 'sklearn.linear_model',
#             algorithm = 'Lasso',
#             transform_method = None),
#         'lasso_lars': borges.TheoryTechnique(
#             name = 'lasso_lars',
#             module = 'sklearn.linear_model',
#             algorithm = 'LassoLars',
#             transform_method = None),
#         'ols': borges.TheoryTechnique(
#             name = 'ols',
#             module = 'sklearn.linear_model',
#             algorithm = 'LinearRegression',
#             transform_method = None),
#         'random_forest': borges.TheoryTechnique(
#             name = 'random_forest',
#             module = 'sklearn.ensemble',
#             algorithm = 'RandomForestRegressor',
#             transform_method = None),
#         'ridge': borges.TheoryTechnique(
#             name = 'ridge',
#             module = 'sklearn.linear_model',
#             algorithm = 'Ridge',
#             transform_method = None),
#         'svm_linear': borges.TheoryTechnique(
#             name = 'svm_linear',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'linear', 'probability': True},
#             transform_method = None),
#         'svm_poly': borges.TheoryTechnique(
#             name = 'svm_poly',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'poly', 'probability': True},
#             transform_method = None),
#         'svm_rbf': borges.TheoryTechnique(
#             name = 'svm_rbf',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'rbf', 'probability': True},
#             transform_method = None),
#         'svm_sigmoid': borges.TheoryTechnique(
#             name = 'svm_sigmoid ',
#             module = 'sklearn.svm',
#             algorithm = 'SVC',
#             required = {'kernel': 'sigmoid', 'probability': True},
#             transform_method = None),
#         'xgboost': borges.TheoryTechnique(
#             name = 'xgboost',
#             module = 'xgboost',
#             algorithm = 'XGBRegressor',
#             # data_dependent = 'scale_pos_weight',
#             transform_method = None)}}

# raw_gpu_options: Dict[str, borges.TheoryTechnique] = {
#     'classify': {
#         'forest_inference': borges.TheoryTechnique(
#             name = 'forest_inference',
#             module = 'cuml',
#             algorithm = 'ForestInference',
#             transform_method = None),
#         'random_forest': borges.TheoryTechnique(
#             name = 'random_forest',
#             module = 'cuml',
#             algorithm = 'RandomForestClassifier',
#             transform_method = None),
#         'logit': borges.TheoryTechnique(
#             name = 'logit',
#             module = 'cuml',
#             algorithm = 'LogisticRegression',
#             transform_method = None)},
#     'cluster': {
#         'dbscan': borges.TheoryTechnique(
#             name = 'dbscan',
#             module = 'cuml',
#             algorithm = 'DBScan',
#             transform_method = None),
#         'kmeans': borges.TheoryTechnique(
#             name = 'kmeans',
#             module = 'cuml',
#             algorithm = 'KMeans',
#             transform_method = None)},
#     'regressor': {
#         'lasso': borges.TheoryTechnique(
#             name = 'lasso',
#             module = 'cuml',
#             algorithm = 'Lasso',
#             transform_method = None),
#         'ols': borges.TheoryTechnique(
#             name = 'ols',
#             module = 'cuml',
#             algorithm = 'LinearRegression',
#             transform_method = None),
#         'ridge': borges.TheoryTechnique(
#             name = 'ridge',
#             module = 'cuml',
#             algorithm = 'RidgeRegression',
#             transform_method = None)}}

# def get_algorithms(settings: Mapping[str, Any]) -> amos.Catalog:
#     """[summary]

#     Args:
#         project (base.Theory): [description]

#     Returns:
#         amos.Catalog: [description]
        
#     """
#     algorithms = raw_options
#     algorithms['model'] = raw_model_options[settings['analyst']['model_type']]
#     if settings['general']['gpu']:
#         algorithms['model'].update(
#             raw_gpu_options[settings['analyst']['model_type']])
#     return algorithms
