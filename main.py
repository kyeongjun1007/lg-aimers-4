from utils import *


def main(config, cbt_params, tuning_params):
    cbt_models = []
    
    for seed in config['seed_list']:
        for tuning_param in tuning_params:
            config['seed'] = seed
            cbt_params['random_seed'] = seed
        
            for param in cbt_params:
                tuning_param[param] = cbt_params[param]
            print(tuning_param)
            
            # seed 설정
            set_seed(config['seed'])

            # 데이터 셋 읽어오기
            df_train, df_test, df_sub = read_data(config)

            # 데이터 전처리
            df_train = feature_engineering(df_train, is_train=True)
            base_num_features, base_cat_features, base_features = make_feature_lists(df_train)

            # 결측치 채우기
            df_train = filling_missing_values(df_train, base_cat_features, base_num_features)

            # 모델 성능 확인(train, valid 데이터 활용)
            cbt_model = model_kfold(df_train, config, tuning_param, base_features, base_cat_features)

            cbt_models.extend(cbt_model)

    # 제출하기
    kfold_submission(df_train, df_test, df_sub, cbt_models)


if __name__ == "__main__":
    config = {
        'train_path': "./Data/train.csv"
        , 'submit_path': "./Data/submission.csv"
        , 'seed_list': [42, 137, 56, 89, 24, 75 ,88 ,36 ,71]
        , 'k_fold': 5
        , 'thresholds': {'product_category': 10, 'expected_timeline': 3}
    }

    cbt_params = {
        'random_seed': config['seed_list'][0],
        'objective': 'Logloss',
        'auto_class_weights': 'Balanced',
        'verbose': 0
    }
    
    
    tuning_params = [
            {
         'learning_rate': 0.05
        , 'n_estimators': 3000

        , 'early_stopping_rounds': 50

        # regularizations
        , 'max_depth': 6
        , 'l2_leaf_reg': 1
        , 'min_data_in_leaf': 2
        , 'subsample': 0.5
        # ,'grow_policy': 'Depthwise' # 'SymmetricTree'(default)
    },
        #    {'early_stopping_rounds': 93, 'learning_rate': 0.07248146664758653, 'n_estimators': 2212, 'max_depth': 6, 'l2_leaf_reg': 18.669558122550747, 'min_data_in_leaf': 4, 'subsample': 0.5995160480057155, 'colsample_bylevel': 0.19993264710979453}
        # ,
        #    {'early_stopping_rounds': 48, 'learning_rate': 0.03671879574710224, 'n_estimators': 1294, 'max_depth': 12, 'l2_leaf_reg': 18.587861100621264, 'min_data_in_leaf': 11, 'subsample': 0.4753484296412652, 'colsample_bylevel': 0.19275904511278602}
        # ,
        #    {'early_stopping_rounds': 83, 'learning_rate': 0.015322722897327458, 'n_estimators': 1006, 'max_depth': 12, 'l2_leaf_reg': 23.064819142140415, 'min_data_in_leaf': 11, 'subsample': 0.4301543072516774, 'colsample_bylevel': 0.18055124326949995}
        # ,
        #    {'early_stopping_rounds': 99, 'learning_rate': 0.06722369315831392, 'n_estimators': 4806, 'max_depth': 6, 'l2_leaf_reg': 17.996700272922553, 'min_data_in_leaf': 4, 'subsample': 0.7944516269381592, 'colsample_bylevel': 0.16510090517729425}
        #  ,
        # {'early_stopping_rounds': 83, 'learning_rate': 0.015322722897327458, 'n_estimators': 1006, 'max_depth': 12, 'l2_leaf_reg': 23.064819142140415, 'min_data_in_leaf': 11, 'subsample': 0.4301543072516774, 'colsample_bylevel': 0.18055124326949995},
        
        # {'early_stopping_rounds': 47, 'learning_rate': 0.020303182253578255, 'n_estimators': 2780, 'max_depth': 12, 'l2_leaf_reg': 17.284664401713627, 'min_data_in_leaf': 15, 'subsample': 0.572162796687132, 'colsample_bylevel': 0.0792062279705981}
    ]

    main(config, cbt_params, tuning_params)

'''
def main():
    seed = 42
    # seed 설정
    set_seed(seed)

    # 데이터 셋 읽어오기
    df_train, df_test, df_sub = read_data()

    # 데이터 전처리
    x_train, x_val, y_train, y_val, df_test = preprocessing(df_train, df_test, seed)

    # 모델 성능 확인(train, valid 데이터 활용)
    model_eval(x_train, x_val, y_train, y_val, seed)

    model = model_train(x_train, x_val, y_train, y_val, seed)

    # 제출하기
    submission(df_test, df_sub, model)
'''