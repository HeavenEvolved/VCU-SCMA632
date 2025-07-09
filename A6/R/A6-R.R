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
