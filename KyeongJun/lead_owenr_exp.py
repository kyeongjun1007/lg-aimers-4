import pandas as pd
import json

df = pd.read_csv('../Data/train.csv')


lead_owner_counts = df['lead_owner'].value_counts()
conversion_rates = df.groupby('lead_owner')['is_converted'].count()

lead_owner_counts = lead_owner_counts.sort_index()
conversion_rates = conversion_rates.sort_index()

df = pd.concat([lead_owner_counts,conversion_rates], axis=1)


alpha_list = [0.5, 1, 1.5, 2]
for alpha in alpha_list:
    df['smoothed_conversion_rate'] = (df['is_converted']+alpha)/(lead_owner_counts + 2*alpha)
    smoothed_conversion_rate = dict(zip(df.index, df['smoothed_conversion_rate']))
    with open(f'../exp/scr_{alpha}.json', 'w') as f:
        json.dump(smoothed_conversion_rate, f)