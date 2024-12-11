import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier

seed = 42
def set_seed(seed: int):
    # random.seed(seed)
    np.random.seed(seed)

df_train = pd.read_csv("train.csv") # 학습용 데이터
df_test = pd.read_csv("submission.csv") # 테스트 데이터(제출파일의 데이터)

def label_encoding(series: pd.Series) -> pd.Series:
    """범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 변환합니다."""

    my_dict = {}

    # 모든 요소를 문자열로 변환
    series = series.astype(str)

    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series

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

df_all = pd.concat([df_train[label_columns], df_test[label_columns]])

for col in label_columns:
    df_all[col] = label_encoding(df_all[col])

for col in label_columns:
    df_train[col] = df_all.iloc[: len(df_train)][col]
    df_test[col] = df_all.iloc[len(df_train) :][col]

x_train, x_val, y_train, y_val = train_test_split(
    df_train.drop("is_converted", axis=1),
    df_train["is_converted"],
    test_size=0.2,
    shuffle=True,
    random_state=seed,
)
def get_clf_eval(y_test, y_pred=None):
    # confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    # print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))

model = BalancedRandomForestClassifier(n_estimators=1000, random_state=seed)

model.fit(x_train.fillna(0), y_train)

for c,v in zip(x_train.columns,model.feature_importances_):
    print(f'{c} : {v}')

# pred = model.predict(x_val.fillna(0))
# get_clf_eval(y_val, pred)
#
# # # 예측에 필요한 데이터 분리
# # x_test = df_test.drop(["is_converted", "id"], axis=1)
# #
# # test_pred = model.predict(x_test.fillna(0))
# # sum(test_pred) # True로 예측된 개수
# #
# # # 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
# # df_sub = pd.read_csv("submission.csv")
# # df_sub["is_converted"] = test_pred
# #
# # # 제출 파일 저장
# # df_sub.to_csv("submission.csv", index=False)