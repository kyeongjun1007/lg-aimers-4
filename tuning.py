import numpy as np
import pandas as pd
import functools

# optuna
import optuna
from optuna.integration import CatBoostPruningCallback

# catboost
from catboost import CatBoostClassifier, Pool

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

from utils import *

def objective(trial: optuna.Trial, config, df, target:str, base_features, base_cat_features) -> float:

    param = {
        'random_seed': config['seed'],
        'objective': 'Logloss',
        # 'class_weights': [1, 9],
        'auto_class_weights': 'Balanced',
        
        # searching parameters
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 5000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.3, 3),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.4, 0.8),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 0.20),
        #'random_strength': trial.suggest_float('random_strength', 1e-2, 20),
        #'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
    }        
        
    # k-fold cross-validation
    skf = StratifiedKFold(n_splits=config['k_fold'], shuffle=True, random_state=config['seed'])
    cbt_models = []
    f1_scores = []

    for k_fold, (train_idx, valid_idx) in enumerate(skf.split(df[base_features], df[target])):
        #print(f'Fold #{k_fold + 1}')
        X_train, y_train = df[base_features].iloc[train_idx], df[target].iloc[train_idx].astype(int)
        X_valid, y_valid = df[base_features].iloc[valid_idx], df[target].iloc[valid_idx].astype(int)
        
        train_pool = Pool(X_train, y_train, cat_features=base_cat_features)
        valid_pool = Pool(X_valid, y_valid, cat_features=base_cat_features)

        cbt = CatBoostClassifier(**param, verbose=0)
        
        # Add a callback for pruning.
        #pruning_callback = CatBoostPruningCallback(trial, 'Logloss')
        cbt.fit(train_pool, 
                eval_set=valid_pool, 
                early_stopping_rounds=param['early_stopping_rounds'], 
                verbose=0, 
                #callbacks=[pruning_callback]
               )
        
        preds = cbt.predict(valid_pool)
        f1_scores.append(f1_score(y_valid, preds))

    return np.mean(f1_scores)

def kfold_tuning(config, n_trials=10):
    # seed 설정
    set_seed(config['seed'])

    # 데이터 셋 읽어오기
    df_train, df_test, df_sub = read_data(config)

     # 데이터 전처리
    df_train = feature_engineering(df_train, is_train=True)
    base_num_features, base_cat_features, base_features = make_feature_lists(df_train)

    # 결측치 채우기
    df_train = filling_missing_values(df_train, base_cat_features, base_num_features)

        
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(
        functools.partial(
            objective,
            config=config,
            df=df_train,
            target='is_converted',
            base_features=base_features,
            base_cat_features=base_cat_features
        ),
        n_trials=n_trials  # Adjust the number of trials as needed
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    config = {
        'train_path': "./Data/train.csv"
        , 'submit_path': "./Data/submission.csv"
        , 'seed': 42
        , 'k_fold': 5
        , 'thresholds': {'product_category': 10, 'expected_timeline': 3}
    }
    
    kfold_tuning(config, n_trials=100)
