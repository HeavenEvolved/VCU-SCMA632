# %% [markdown]
# # **Imports**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats import weightstats as stats 

# %% [markdown]
# # **Set Paths**

# %%
import os
BASE = os.getcwd()
DATA = os.path.join(BASE, 'datasets')
print(BASE, DATA)

# %% [markdown]
# # **Load Main Dataset**

# %%
data_file_name = 'NSSO68.csv'
data_file = os.path.join(DATA, data_file_name)

df = pd.read_csv(data_file, encoding='latin1', low_memory=False, index_col='slno')

# %%
df.head()

# %% [markdown]
# # **Drop Unnecessary Attributes**

# %%
# Keeping only the relevant columns
relevant_columns = [
    'state',
    'state_1', 
    'District', 
    'Region', 
    'Sector', 
    'State_Region', 
    'Meals_At_Home', 
    'ricetotal_v', 
    'wheattotal_v', 
    'Milktotal_v', 
    'pulsestot_v', 
    'nonvegtotal_v', 
    'fruitstt_v', 
    'No_of_Meals_per_day'
]

df.drop(columns=[col for col in df.columns if col not in relevant_columns], inplace=True)

# %% [markdown]
# # **Filter for assigned State**

# %%
# MEGHALAYA State Code: 17
df_meghalaya = df[df['state'] == 17]
del df

# %%
df_meghalaya.head()

# %% [markdown]
# # **Save Filtered Dataset**

# %%
df_meghalaya.reset_index(inplace=True)
df_meghalaya.drop(columns=['slno'], inplace=True)
df_meghalaya.index.name = 'slno'

my_data_file_name = 'meghalaya_NSSO68.csv'
df_meghalaya.to_csv(os.path.join(DATA, my_data_file_name))

# %%
df_meghalaya.info()

# %% [markdown]
# # **Impute Null Values**

# %%
for column in df_meghalaya.columns:
    null_c = df_meghalaya[column].isnull().sum()
    if null_c > 0:
        df_meghalaya[column] = df_meghalaya[column].fillna(df_meghalaya[column].mean())
        print(f"{column}: {null_c} null values")

# %%
df_meghalaya.info()

# %% [markdown]
# # **Handle Outliers in Consumption Columns**

# %%
consumption_columns = [
    'ricetotal_v', 
    'wheattotal_v', 
    'Milktotal_v', 
    'pulsestot_v', 
    'nonvegtotal_v', 
    'fruitstt_v'
]

# %%
def check_outliers():
    # Create subplots to show boxplots for each consumption column
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))

    for ax, column in zip(axes.flatten(), consumption_columns):
        df_meghalaya.boxplot(column=column, ax=ax)
        ax.set_title(f'Boxplot of {column}')
        ax.set_ylabel('Consumption (kg/month)')
        ax.set_xlabel('')
        
    plt.tight_layout()

# %%
def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# %%
check_outliers()

# %%
for column in consumption_columns:
    df_meghalaya = remove_outliers(df_meghalaya, column)

# %%
check_outliers()

# %% [markdown]
# # **Create Total Consumption Column**

# %%
df_meghalaya['total_consumption'] = df_meghalaya[consumption_columns].sum(axis=1)

# %% [markdown]
# # **Descriptive Statistics for Total Consumption by District, Region and Sector**

# %%
def summarize_consumption(df, column):
    summary = df.groupby(column)['total_consumption'].agg(['mean', 'std', 'count', 'sum'])
    summary.sort_values(by='sum', ascending=False, inplace=True)
    return summary

# %%
district = summarize_consumption(df_meghalaya, 'District')
region = summarize_consumption(df_meghalaya, 'Region')
sector = summarize_consumption(df_meghalaya, 'Sector')

# %%
district.head()

# %%
district.tail()

# %% [markdown]
# # **Map District and Sector to their Names/Labels**

# %%
dist_st_map = {
    1: 'West Garo Hills',
    2: 'East Garo Hills',
    3: 'South Garo Hills',
    4: 'West Khasi Hills',
    5: 'Ri Bhoi',
    6: 'East Khasi Hills',
    7: 'Jaintia Hills',
}

sector_map = {
    1: 'Rural',
    2: 'Urban'
}

df_meghalaya['District'] = df_meghalaya['District'].map(dist_st_map)
df_meghalaya['Sector'] = df_meghalaya['Sector'].map(sector_map)

# %%
district = summarize_consumption(df_meghalaya, 'District')
region = summarize_consumption(df_meghalaya, 'Region')
sector = summarize_consumption(df_meghalaya, 'Sector')

# %%
district

# %% [markdown]
# # **Hypothesis Testing using Z-Test as n > 30**

# %%
urban = df_meghalaya[df_meghalaya['Sector'] == 'Urban']['total_consumption']
rural = df_meghalaya[df_meghalaya['Sector'] == 'Rural']['total_consumption']

sector_z_stat, sector_p_value = stats.ztest(urban, rural, alternative='two-sided')

# %%
print(sector_z_stat)
print(sector_p_value)

# %%
if sector_p_value < 0.05:
    print("There is a significant difference in total consumption between Urban and Rural sectors. Reject the null hypothesis.")
else:
    print("There is no significant difference in total consumption between Urban and Rural sectors. Failed to reject the null hypothesis.")

# %%
top_dist = df_meghalaya[df_meghalaya['District'] == district.iloc[0].name]['total_consumption']
bottom_dist = df_meghalaya[df_meghalaya['District'] == district.iloc[-1].name]['total_consumption']

dist_z_stat, dist_p_value = stats.ztest(top_dist, bottom_dist, alternative='two-sided')

# %%
print(dist_z_stat)
print(dist_p_value)

# %%
if dist_p_value < 0.05:
    print("There is a significant difference in total consumption between the top and bottom districts. Reject the null hypothesis.")
else:
    print("There is no significant difference in total consumption between the top and bottom districts. Failed to reject the null hypothesis.")


