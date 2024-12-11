# lg-aimers-4
## 개요
주최 : LG AI 연구원   
주제 : B2B 마케팅 (견적서 데이터를 활용한 고객 구매 전환 분류 모델링)   
기간 : 2024.02.01 - 2024.02.28 (예선) / 2024.04.06 - 2024.04.07 (본선)   
설명   
As-is : 영업사원이 직접 고객 데이터를 바탕으로 영업 대상 선정   
To-be : 예측 모델을 통해 구매 전환율이 높은 고객을 추출하여 영업사원에게 추천   
   
__자세한 분석 과정에 대한 설명은 ['발표자료.pdf'](./발표자료.pdf) 파일 참고!__   
__실제 개발 코드는 main.py, utils.py 참고!__

## 파일 설명
lg-aimers-4   
ㄴ /Name Folders (ex. KyeongJun) : 팀원 별 EDA, 분석 환경   
ㄴ /visual_func : 본선 과정 중 빠른 시각화를 위한 함수 코드 미리 작성   
ㄴ LG_Aimers_4th.ipynb : 제출 파일 (CatBoost, Seed Ensemble, CrossValidation)   
ㄴ main.py : main 실행 함수 (seed, hyper-parameter 설정)   
ㄴ utils.py : 기능 모듈화 (pre-processing, training, validation etc)   
ㄴ tuning.py : Hyper-parameter tuning을 위한 코드   
ㄴ 발표자료.pdf : 분석 내용 발표 자료   
ㄴ requirements.yml   
