import pandas as pd

df = pd.read_csv('data/real_training_data_v3.csv')

bad_dates = ['2022-03-28', '2022-04-10', '2022-04-25']
mask = df['date'].isin(bad_dates)

df.loc[mask, 'ndwi'] = -0.1
df.loc[mask, 'ndwi_real'] = False
df.loc[mask, 'data_quality'] = df.loc[mask, 'data_quality'].replace('full_real', 'sar_only')

df.to_csv('data/real_training_data_v3.csv', index=False)

print("✅ Fixed NDWI for bad dates successfully!")
print(df[df['date'].isin(bad_dates)][['date', 'ndwi', 'ndwi_real', 'data_quality']])
print("Done!")