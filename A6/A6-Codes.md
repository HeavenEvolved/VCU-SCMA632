# **Codes**

## **Python Code (Exported from IPYNB)**

```python
# %% [markdown]
# # **Imports**

# %%
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

# %% [markdown]
# # **Read Dataset**

# %%
# Reading the CSV file
BASE = os.getcwd()

DATASETS_DIR = os.path.join(BASE, "datasets")
DATASET_FILE = "NSSO68.csv"
DATASET_PATH = os.path.join(DATASETS_DIR, DATASET_FILE)

df = pd.read_csv(DATASET_PATH, low_memory=False)

# %%
df.head()

# %%
# Filtering for Meghalaya Data
print((df['state'] == 17).sum())

# %%
meg_df = df[df['state'] == 17].copy()

# %% [markdown]
# # **Plot Histogram of Food Consumption in Meghalaya**

# %%
plt.hist(meg_df['foodtotal_v'].dropna(), bins=30, color='red', edgecolor='white')
plt.title("Distribution of Food Consumption in Meghalaya")
plt.xlabel("Food Consumption (foodtotal_v)")
plt.ylabel("Frequency")
plt.show()

# %%
meg_df['District'] = meg_df['District'].astype(str).str.zfill(2)
meg_df['DWCons'] = meg_df.groupby('District')['foodtotal_v'].transform('mean')

# %%
district_map = pd.DataFrame({
    'DistrictCode': [f"{i:02d}" for i in range(1, 8)],
    'DistrictName': [
        "West Garo Hills",
        "East Garo Hills",
        "South Garo Hills",
        "West Khasi Hills",
        "Ribhoi",
        "East Khasi Hills",
        "Jaintia Hills"
    ]
})

# %%
meg_df['DistrictCode'] = meg_df['District']
meg_df = meg_df.merge(district_map, on='DistrictCode', how='left')

# %% [markdown]
# # **Plot Bar graph for Avg Food Consumption by District**

# %%
district_avg = meg_df.groupby('DistrictName')['foodtotal_v'].mean().reset_index()
district_avg = district_avg.sort_values(by='foodtotal_v', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(district_avg['DistrictName'], district_avg['foodtotal_v'], color='skyblue')
plt.xticks(rotation=90)
plt.title("Average Food Consumption by District (Meghalaya)")
plt.ylabel("Average Food Consumption (Rs.)")
plt.tight_layout()
plt.show()

# %% [markdown]
# # **Display Map of Districts in Meghalaya overlaid with Avg Food Consumption**

# %%
GEOJSON_FILE = "MEGHALAYA_DISTRICTS.geojson"
GEOJSON_PATH = os.path.join(DATASETS_DIR, GEOJSON_FILE)

# Read GeoJSON file
data_map = gpd.read_file(GEOJSON_PATH)

# Check and rename the district column
data_map = data_map.rename(columns={'dtname': 'DistrictName'})

# Merge geo data with average food data
data_map_data = data_map.merge(district_avg, on='DistrictName', how='left')

print(data_map_data.columns.tolist())

# Fill missing values with 0
data_map_data['foodtotal_v'] = data_map_data['foodtotal_v'].fillna(0)

# Plot choropleth map
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
data_map_data.plot(column='foodtotal_v',
                   cmap='YlOrRd',
                   linewidth=0.8,
                   ax=ax,
                   edgecolor='0.8',
                   legend=True)

# Add labels
for idx, row in data_map_data.iterrows():
    if row['geometry'].centroid.is_empty:
        continue
    plt.annotate(text='{} - {}'.format(row['DistrictName'], round(row['foodtotal_v'], 2)),
                 xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
                 horizontalalignment='center',
                 fontsize=8,
                 color='black')

plt.title("Average Food Consumption by District (Meghalaya)", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
```

## **R Code**

```r
BASE <- getwd()

DATASET_DIR <- paste(BASE, "/datasets", sep = "")
DATASET_FILE <- "NSSO68.csv"
DATASET_PATH <- paste(DATASET_DIR, "/", DATASET_FILE, sep = "")

df <- read.csv(DATASET_PATH)

head(df$foodtotal_v)

sum(df$state_1 == "MEG")

meg_df <- df[df$state_1 == "MEG", ]
dim(meg_df)

hist(
  meg_df$foodtotal_v,
  main = "Distribution of Food Consumption in Meghalaya",
  xlab = "Food Consumption (foodtotal_v)",
  col = "lightblue",
  border = "white"
)

meg_df$District <- as.factor(meg_df$District)

library(dplyr)

meg_df <- meg_df %>%
  group_by(District) %>%
  mutate(DWCons = mean(foodtotal_v, na.rm = TRUE)) %>%
  ungroup()

district_map <- data.frame(
  DistrictCode = sprintf("%02d", 1:7),
  DistrictName = c(
    "West Garo Hills",
    "East Garo Hills",
    "South Garo Hills",
    "West Khasi Hills",
    "Ribhoi",
    "East Khasi Hills",
    "Jaintia Hills"
  ),
  stringsAsFactors = FALSE
)

meg_df <- meg_df %>%
  mutate(DistrictCode = sprintf("%02d", as.numeric(District)))

meg_df <- meg_df %>%
  left_join(district_map, by = "DistrictCode")

district_avg <- meg_df %>%
  group_by(DistrictName) %>%
  summarise(avg_food = mean(foodtotal_v, na.rm = TRUE)) %>%
  arrange(desc(avg_food))

barplot(
  height = district_avg$avg_food,
  names.arg = district_avg$DistrictName,
  las = 2,
  col = "skyblue",
  main = "Average Food Consumption by District (Meghalaya)",
  ylab = "Avgerage Food Consumption (Rs.)",
  cex.names = 0.7
)

library(ggplot2)
library(sf)
library(dplyr)

GEOJSON_FILE <- "MEGHALAYA_DISTRICTS.geojson"
GEOJSON_PATH <- paste(DATASET_DIR, "/", GEOJSON_FILE, sep = "")

data_map <- st_read(GEOJSON_PATH)

data_map <- data_map %>%
  rename(DistrictName = dtname)

data_map_data <- data_map %>%
  left_join(district_avg, by = "DistrictName")

data_map_data$avg_food[is.na(data_map_data$avg_food)] <- 0

ggplot(data_map_data) +
  geom_sf(aes(fill = avg_food, geometry = geometry)) +
  scale_fill_gradient(low = "yellow", high = "red") +
  ggtitle("Average Food Consumption by District (Meghalaya)") +
  theme_minimal() +
  geom_sf_text(aes(label = paste(DistrictName, "-", round(avg_food, 2))), size = 3, color = "black")
```
