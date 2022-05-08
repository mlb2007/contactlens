#%%

import pandas as pd
import random
seed = 92

# read data
df = pd.read_csv("../data/10k_sent_2019.csv", index_col=0)
df

# %%

# randomly sample firms
all_firms = list(df.groupby("cik").size().index)
print(f"Number of firms: {len(all_firms)}")
sample_idxs = random.sample(range(len(all_firms)), 2500)
sample_firms = [all_firms[i] for i in sample_idxs]
print(f"Number of sampled firms: {len(sample_firms)}")

#%%

# sample complete data and save it as parquet
df_sample = df.loc[df["cik"].isin(sample_firms)]
df_sample.reset_index(drop=True, inplace=True)
df_sample.to_parquet('../data/10k_sent_2019_sample.parquet')
# %%
