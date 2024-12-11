import random
import pandas as pd
import numpy as np
from datetime import datetime
import os

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier
from functools import partial


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    # torch.backends.cudnn.benchmark = True  # type: ignore


def read_data(config):
    df_train = pd.read_csv(config['train_path'])  # 학습용 데이터
    df_test = pd.read_csv(config['submit_path'])  # 테스트 데이터(제출파일의 데이터)
    df_sub = pd.read_csv(config['submit_path'])

    return df_train, df_test, df_sub


def get_clf_eval(y_test, y_pred=None, fold_no=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    fold_info = f'Fold #{fold_no}' if fold_no is not None else ''
    print(f'{fold_info} ACC: {accuracy:.4f}, PRE: {precision:.4f}, REC: {recall:.4f}, F1: {F1:.4f}\n')
    return F1


def new_business_area(cur_area):
    if cur_area in ['corporate / office', 'government department']:
        return 'Office'
    elif cur_area in ['education', 'public facility']:
        return 'Public'
    elif cur_area in ['hotel & accommodation', 'residential (home)']:
        return 'Amenity'
    elif cur_area in ['factory', 'power plant / renewable energy', 'transportation']:
        return 'Industry'
    else:
        return cur_area


def make_value_count(value_count_dict, val):
    if val in value_count_dict:
        return value_count_dict[val]
    return np.NAN


def make_value_count_dict(df, col_names):
    value_count_dict = dict()
    for feat_name in col_names:
        total_count = df[feat_name].value_counts()

        count_df = pd.DataFrame(total_count).reset_index(drop=False)

        value_count_dict[feat_name] = dict(zip(count_df.iloc[:, 0], count_df.iloc[:, 1]))

    return value_count_dict


def make_continent(rc):
    if rc in ('lgein', 'lgeml', 'lgeph', 'lgeth', 'lgevh', 'lgeil', 'lgekr', 'lgett', 'lgejp', 'lgech', 'lgeir', 'lgesj',
              'lgegf', 'lgetk', 'lgelf', 'lgehk', 'lgeyk'):
        return 'asia'
    if rc in ('lgeaf', 'lgesa', 'lgemc', 'lgeas', 'lgeeg', 'lgeef'):
        return 'africa'
    if rc in ('lgeus', 'lgeci'):
        return 'northamerica'
    if rc in ('lgesp', 'lgecb', 'lgems', 'lgecl', 'lgeps', 'lgear', 'lgepr'):
        return 'southamerica'
    if rc in ('lgeuk', 'lgees', 'lgefs', 'lgebn', 'lgebt', 'lgedg', 'lgero', 'lgemk', 'lgepl', 'lgecz', 'lgehs', 'lgesw',
              'lgeag', 'lgeeb', 'lgera', 'lgeur', 'lgept', 'lgeis', 'lgela'):
        return 'europe'
    if rc in ('lgesl', 'lgeap'):
        return 'oceania'
    return np.NaN


def parse_budget(budget):
    #print('current:', budget)
    
    if pd.isna(budget):
        return np.nan, np.nan
    
    if "less than" in budget.lower():
        upper_limit = int(''.join(filter(str.isdigit, budget.split("less than")[1])))
        return 0, upper_limit
    elif "more than" in budget.lower():
        lower_limit = int(''.join(filter(str.isdigit, budget.split("more than")[1])))
        return lower_limit, 999999999
    elif "~" in budget or "-" in budget or "to" in budget or "y" in budget:
        parts = budget.replace('~', '-').replace('to', '-').replace('y', '-').split('-')
        try:
            lower_limit = int(''.join(filter(str.isdigit, parts[0])))
            upper_limit = int(''.join(filter(str.isdigit, parts[1])))
            return lower_limit, upper_limit
        except:
            return np.nan, np.nan
    else:  # For cases with single values like '25000 USD', '5000', etc.
        try:
            amount = int(''.join(filter(str.isdigit, budget)))
            return amount, amount
        except:
            return np.nan, np.nan
        
def map_currency(value):
    value = str(value).lower()
    if '$' in value:
        return 'USD'
    elif '€' in value:
        return 'EUR'
    elif '£' in value:
        return 'GBP'
    elif 'cop' in value:
        return 'COP'
    elif 'idr' in value:
        return 'IDR'
    else:
        return np.nan
    
    
def convert_to_usd(row, lower_or_upper_key):
    conversion_rates = {
        'GBP': 1.25,
        'EUR': 1.08,
        'COP': 0.00027,  # Assuming conversion rate is per 1 unit of currency to USD
        'USD': 1  # USD to USD conversion is 1:1
    }
    
    # Check if the currency column or the budget value is NaN
    if pd.isna(row['currency']) or pd.isna(row[lower_or_upper_key]):
        return np.nan  # Return NaN if there's no currency info or the budget value is missing
    
    # Convert the budget to USD based on the currency
    rate = conversion_rates.get(row['currency'], 1)  # Default to 1 if the currency is not recognized
    return row[lower_or_upper_key] * rate


def feature_engineering(df_input, is_train=False):
    df = df_input.copy()

    # drop_duplicates
    df = df.drop_duplicates(keep='first')

    # converting text to lowercase, removing commas and periods
    # and replacing underscores and hyphens with spaces.
    for feat in df.columns:
        # Applies transformations only to string-type columns.
        if df[feat].dtype == 'object' and feat != 'lead_date':
            df[feat] = df[feat].str.lower()
            df[feat] = df[feat].str.replace('[,\.]', '', regex=True)
            df[feat] = df[feat].str.replace('[_-]', ' ', regex=True)

    # Recategorizes 'Solution' and 'CM' in 'business_unit' column to 'Others'.
    df['business_unit'] = np.where(df['business_unit'].isin(['Solution', 'CM']), 'Others', df['business_unit'])

    # map ['other', 'others', 'etc'] -> 'Others'
    unify_others_columns = ['customer_type', 'customer_job', 'inquiry_type', 'product_category', 'product_subcategory',
                            'customer_position', 'expected_timeline']
    for column in unify_others_columns:
        df[column] = df[column].replace(['other', 'others', 'etc'], 'others')

    # make new columns
    df['continent'] = df['response_corporate'].map(make_continent)
    df['business_area_group'] = df['business_area'].map(new_business_area)
#     df['product_category_count'] = df['product_category'].apply(
#         lambda x: len(str(x).split(',')) if not pd.isna(x) else np.nan)
    
    ### 본선 data 전처리
    df[['year', 'month', 'day']] = df['lead_date'].str.split('-', expand=True)

    # 'date' 컬럼을 datetime 형식으로 변환
    df['lead_date'] = pd.to_datetime(df['lead_date'])

    # 요일 추출
    df['day_of_week'] = df['lead_date'].dt.day_name()

    df['quarter'] = df['month'].apply(lambda x: (int(x) - 1) // 3 + 1)
    
    df['season'] = df['month'].apply(lambda x: (int(x) - 3) // 3 + 1 if int(x) >= 3 else 4)
    
    # 주중, 주말
    df['weekday'] = df['day_of_week'].apply(lambda x: 0 if x in ['Saturday', 'Sunday'] else 1)
    
#     df['area_rate'] = df['business_area'] + ' ' + df['ver_win_rate_x'].astype(str)
    
    make_value_count_columns = ['lead_owner']
    if is_train:
        global value_count_dict
        value_count_dict = make_value_count_dict(df_input, make_value_count_columns)

    for feat_name in value_count_dict:
        func = partial(make_value_count, value_count_dict[feat_name])
        df[f'{feat_name}_count'] = df[feat_name].map(func)

    # correct data type
    for feat in ['customer_idx', 'lead_owner', 'ver_cus', 'ver_pro', 'weekday', 'season']:
        df[feat] = df[feat].astype(object)
    
    # expected_budget
    # -> budget_prep, currency, lower_budget, lower_budget_usd, upper_budget, upper_budget_usd
    df['budget_prep'] = df['expected_budget'].str.replace('_', ' ', regex=True)
    df['budget_prep'] = df['budget_prep'].str.replace('weniger als', 'less than', case=False)
    df['budget_prep'] = df['budget_prep'].str.replace('menos de', 'less than', case=False)
    df['budget_prep'] = df['budget_prep'].str.replace('über', 'more than', case=False)
    df['budget_prep'] = df['budget_prep'].str.replace('less than', 'less than', case=False)
    df['budget_prep'] = df['budget_prep'].str.replace('more than', 'more than', case=False)
    df['budget_prep'] = df['budget_prep'].str.replace('.', ',')
    
    df['currency'] = df['budget_prep'].apply(map_currency)
    df[['lower_budget', 'upper_budget']] = df['budget_prep'].apply(lambda x: pd.Series(parse_budget(x)))
    df['lower_budget_usd'] = df.apply(lambda row: convert_to_usd(row, 'lower_budget'), axis=1)
    df['upper_budget_usd'] = df.apply(lambda row: convert_to_usd(row, 'upper_budget'), axis=1)
    
    df['lower_budget_usd'] = df['lower_budget_usd'].fillna(-1)
    df['upper_budget_usd'] = df['upper_budget_usd'].fillna(999999999)
    
    # only use currency, lower_budget_usd, upper_budget_usd
    df = df.drop(['lower_budget', 'upper_budget', 'lead_desc_length'], axis=1)
#     df = df.drop(['expected_budget', 'lower_budget', 'upper_budget'], axis=1)
    
    # lead desc length 재정의
    df['lead_desc_length'] = df['lead_description'].apply(lambda x: len(x) if type(x) != float else np.nan)
#     df['lead_desc_length'] = df['lead_description'].apply(lambda x: len(x.split()) if type(x) != float else np.nan)
    return df


def make_feature_lists(df):
    base_features = []     # all features except target variable.
    base_num_features = [] # numerical features
    base_cat_features = [] # categorical features

    for feat in df.columns:
        # skip the target
        if feat == 'is_converted':
            continue

        base_features.append(feat)

        if df[feat].dtype == 'object':
            base_cat_features.append(feat)
        else:
            base_num_features.append(feat)

    # features to be removed from data analysis
    removal_features = {
        'id', 'bant_submit', 'id_strategic_ver', 'it_strategic_ver', 'idit_strategic_ver',
        'customer_country', 'customer_country.1', 'business_subarea', 'business_area',
      'lead_date', 'day', 'lead_description', 'day_of_week', 'weekday',
        'currency', 'lower_budget_usd', 'upper_budget_usd'
    }

    # remove the specified features
    base_num_features = [i for i in base_num_features if i not in removal_features]
    base_cat_features = [i for i in base_cat_features if i not in removal_features]
    base_features = [i for i in base_features if i not in removal_features]
    
    return base_num_features, base_cat_features, base_features

def filling_missing_values(df_input, base_cat_features, base_num_features):
    df = df_input.copy()

    # Fill missing values for categorical features with 'UNK'
    # and ensure their data type is string.
    for base_cat_feat in base_cat_features:
        df[base_cat_feat] = df[base_cat_feat].fillna('UNK')
        df[base_cat_feat] = df[base_cat_feat].astype(str)

    # Fill missing values for numerical features with -1.
    for base_num_feat in base_num_features:
        df[base_num_feat] = df[base_num_feat].fillna(-1)

    return df

def model_kfold(df, config, cbt_params, base_features, base_cat_features):
    target = 'is_converted'

    skf = StratifiedKFold(n_splits=config['k_fold'], shuffle=True, random_state=config['seed'])
    cbt_models = []
    f1_scores = []

    for k_fold, (train_idx, valid_idx) in enumerate(skf.split(df[base_features], df[target])):
        print(f'Fold #{k_fold + 1}')
        X_train, y_train = df[base_features].iloc[train_idx], df[target].iloc[train_idx].astype(int)
        X_valid, y_valid = df[base_features].iloc[valid_idx], df[target].iloc[valid_idx].astype(int)

        cbt = CatBoostClassifier(**cbt_params)

        cbt.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            cat_features=base_cat_features,
        )

        cbt_models.append(cbt)

        # train-set
        print('[Train] ', end='')
        y_pred = cbt.predict(X_train)
        _ = get_clf_eval(y_train, y_pred, k_fold + 1)

        # valid-set
        print('[Valid] ', end='')
        y_pred = cbt.predict(X_valid)
        y_pred = y_pred.astype(y_valid.dtype)
        f1 = get_clf_eval(y_valid, y_pred, k_fold + 1)

        f1_scores.append(f1)

    print(f'Avg. F1 of validset: {np.mean(f1_scores)}')
    print(f'Var. F1 of validset: {np.var(f1_scores)}')

    return cbt_models


def kfold_submission(df_train, df_test, df_sub, cbt_models):
    folder_path = 'FeatureImportance'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 현재 날짜와 시간을 얻습니다.
    now = datetime.now()

    # 년월일시분을 기록합니다.
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    submission_time = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"[2:]
    target = 'is_converted'

    df_test = feature_engineering(df_test)
    base_num_features, base_cat_features, base_features = make_feature_lists(df_test)

    df_test = filling_missing_values(df_test, base_cat_features, base_num_features)

    X_test = df_test[base_features]

    y_probs = np.zeros((X_test.shape[0], 2))
    df_feature_importance_all = pd.DataFrame({'features': base_features})

    for i, cbt_model in enumerate(cbt_models):
        y_probs += cbt_model.predict_proba(X_test) / len(cbt_models)

        df_feature_importance_all[f'model_{i}'] = cbt_model.get_feature_importance()

    df_feature_importance_all['average'] = df_feature_importance_all.iloc[:, 1:].mean(axis=1).values
    df_feature_importance_all['rank'] = df_feature_importance_all['average'].rank(ascending=False)

    df_feature_importance_all.to_csv(f'./FeatureImportance/feat_import_{submission_time}.csv', index=False)

    # 제출 파일 작성
    df_sub[target] = (y_probs[:, 1] >= 0.5).astype(bool)
    print(y_probs[:, 1])

    # 제출 파일 저장
    df_sub.to_csv(f"./Data/submission_{submission_time}.csv", index=False)


'''
def label_encoding(series: pd.Series) -> pd.Series:
    """범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 변환합니다."""

    my_dict = {}

    # 모든 요소를 문자열로 변환
    series = series.astype(str)

    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series

def new_customer_type(cur_type):
    if cur_type in ['End Customer']:
        return 'End-Customer'
    elif cur_type in ['Commercial end-user']:
        return 'End-user'
    elif cur_type in ['Specifier/ Influencer']:
        return 'Specifier / Influencer'
    elif cur_type in ['Software/Solution Provider']:
        return 'Software / Solution Provider'
    elif cur_type in ['Homeowner']:
        return 'Home Owner'
    elif cur_type in ['Etc.', 'Others']:
        return 'Other'
    else:
        return cur_type

def make_weighted_mean_dict(df, col_names):
    weighted_mean_dict = dict()
    for feat_name in col_names:
        true_val = df[df['is_converted']][feat_name].mean()
        false_val = df[~df['is_converted']][feat_name].mean()
        true_ratio = df[df[feat_name].isna()]['is_converted'].mean()

        weighted_mean = true_ratio * true_val + (1-true_ratio) * false_val
        weighted_mean_dict[feat_name] = weighted_mean

    return weighted_mean_dict

def preprocessing(df_train, df_test, seed):
    # 레이블 인코딩할 칼럼들
    label_columns = [
        "customer_country",
        "business_subarea",
        "business_area",
        "business_unit",
        "customer_type",
        "enterprise",
        "customer_job",
        "inquiry_type",
        "product_category",
        "product_subcategory",
        "product_modelname",
        "customer_country.1",
        "customer_position",
        "response_corporate",
        "expected_timeline",
    ]

    # train test concat
    df_all = pd.concat([df_train[label_columns], df_test[label_columns]])

    # label encoding
    for col in label_columns:
        df_all[col] = label_encoding(df_all[col])

    # train test split
    for col in label_columns:
        df_train[col] = df_all.iloc[: len(df_train)][col]
        df_test[col] = df_all.iloc[len(df_train):][col]

    # 학습, 검증 데이터 분리
    x_train, x_val, y_train, y_val = train_test_split(
        df_train.drop("is_converted", axis=1),
        df_train["is_converted"],
        test_size=0.2,
        shuffle=True,
        random_state=seed,
        stratify=df_train["is_converted"],
    )
    return x_train, x_val, y_train, y_val, df_test

def model_eval(x_train, x_val, y_train, y_val, seed):
    # 모델 정의 (향후 argument 추가 예정)

    # model = DecisionTreeClassifier(random_state = seed)
    # model = RandomForestClassifier(random_state = seed)
    # model = ExtraTreesClassifier(random_state=seed)
    # model = CatBoostClassifier(random_state=seed)
    model = LGBMClassifier(random_state=seed)
    # model = XGBClassifier(random_state=seed)

    # 모델 학습
    model.fit(x_train.fillna(0), y_train)

    # 모델 성능 보기
    pred = model.predict(x_val.fillna(0))
    get_clf_eval(y_val, pred)

    # return model


def model_train(x_train, x_val, y_train, y_val, seed):
    # 모델 정의 (향후 argument 추가 예정)

    # model = DecisionTreeClassifier(random_state = seed)
    # model = RandomForestClassifier(random_state = seed)
    # model = ExtraTreesClassifier(random_state=seed)
    # model = CatBoostClassifier(random_state=seed)
    model = LGBMClassifier(random_state=seed)
    # model = XGBClassifier(random_state=seed)

    # train valid concat

    x, y = pd.concat([x_train, x_val], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True)

    # 모델 학습
    model.fit(x.fillna(0), y)
    return model

def submission(df_test, df_sub, model):
    # 현재 날짜와 시간을 얻습니다.
    now = datetime.now()

    # 년월일시분을 기록합니다.
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    submission_time = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"[2:]

    # 예측에 필요한 데이터 분리
    x_test = df_test.drop(["is_converted", "id"], axis=1)

    test_pred = model.predict(x_test.fillna(0))

    # 제출 파일 작성
    df_sub["is_converted"] = test_pred

    # 제출 파일 저장
    df_sub.to_csv(f"./Data/submission_{submission_time}.csv", index=False)

def feature_engineering(df_input, is_train=False):
    df = df_input.copy()

    # lead_owner_count 변수 생성 (dict 생성)
    make_value_count_columns = ['lead_owner']

    if is_train:
        global value_count_dict
        value_count_dict = make_value_count_dict(df, make_value_count_columns)

    # drop_duplicates
    df = df.drop_duplicates(keep='first')

    # categorical variables pre-processing
    df['continent'] = df['response_corporate'].map(make_continent)

    for feat in df.columns:  # text : 소문자 통일, [쉼표, 마침표]제거, [밑줄, 하이픈]공백으로 변환
        if df[feat].dtype == 'object':
            df[feat] = df[feat].str.lower()
            df[feat] = df[feat].str.replace('[,\.]', '', regex=True)
            df[feat] = df[feat].str.replace('[_-]', ' ', regex=True)
    df['business_unit'] = np.where(df['business_unit'].isin(['Solution', 'CM']), 'Others', df['business_unit'])
    df['business_area_group'] = df['business_area'].map(new_business_area)

    #'other', 'others', 'etc' 값을 'Others'로 통일
    unify_others_columns = ['customer_type', 'customer_job', 'inquiry_type', 'product_category', 'product_subcategory', 'customer_position', 'expected_timeline']
    for column in unify_others_columns:
        df[column] = df[column].replace(['other', 'others', 'etc'], 'others')

    df['product_category_count'] = df['product_category'].apply(lambda x: len(str(x).split(',')) if not pd.isna(x) else np.nan)

    # lead_owner_count 변수 생성
    for feat_name in value_count_dict:
        func = partial(make_value_count, value_count_dict[feat_name])
        df[f'{feat_name}_count'] = df[feat_name].map(func)

    # features list
    # 변수 삭제
    # object로 변경
    base_features = []
    base_num_features = []
    base_cat_features = []

    for feat in df.columns:
        if feat == 'is_converted':
            continue

        base_features.append(feat)

        # Get the data type of the column
        if feat in ['customer_idx', 'lead_owner', 'ver_cus', 'ver_pro']:
            df[feat] = df[feat].astype(object)

        if df[feat].dtype == 'object':
            base_cat_features.append(feat)
        else:
            base_num_features.append(feat)

    removal_features = {
        'id', 'bant_submit', 'id_strategic_ver', 'it_strategic_ver', 'idit_strategic_ver',
        'customer_country', 'customer_country.1', 'business_subarea', 'business_area'
    }

    base_num_features = [i for i in base_num_features if i not in removal_features]
    base_cat_features = [i for i in base_cat_features if i not in removal_features]
    base_features = [i for i in base_features if i not in removal_features]

    return df, base_num_features, base_cat_features, base_features
'''