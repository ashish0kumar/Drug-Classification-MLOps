# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# Load dataset
drug_data <- read.csv("D:\\ML-Devops\\Data\\drug.csv")

# View structure and summary
str(drug_data)
summary(drug_data)

# Check for missing values
colSums(is.na(drug_data))

# Convert categorical variables to factors
drug_data$Sex <- as.factor(drug_data$Sex)
drug_data$BP <- as.factor(drug_data$BP)
drug_data$Cholesterol <- as.factor(drug_data$Cholesterol)
drug_data$Drug <- as.factor(drug_data$Drug)

# Plot: Drug distribution
ggplot(drug_data, aes(x = Drug, fill = Drug)) +
  geom_bar() +
  theme_minimal() +
  ggtitle("Distribution of Drug Types")

# Plot: Age distribution by Drug
ggplot(drug_data, aes(x = Drug, y = Age, fill = Drug)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Age distribution for each Drug")

# Plot: Na_to_K distribution by Drug
ggplot(drug_data, aes(x = Drug, y = Na_to_K, fill = Drug)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Na_to_K ratio distribution by Drug")

# Plot: Drug vs BP
ggplot(drug_data, aes(x = BP, fill = Drug)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  ggtitle("BP levels across Drug types")

# Plot: Drug vs Cholesterol
ggplot(drug_data, aes(x = Cholesterol, fill = Drug)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  ggtitle("Cholesterol levels across Drug types")

# Plot: Drug vs Sex
ggplot(drug_data, aes(x = Sex, fill = Drug)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  ggtitle("Sex distribution across Drug types")

# Optional: Encode factors into numeric (if needed for ML)
drug_data_encoded <- drug_data %>%
  mutate(
    Sex = as.numeric(Sex),
    BP = as.numeric(BP),
    Cholesterol = as.numeric(Cholesterol),
    Drug = as.numeric(Drug)
  )

# Save preprocessed data
write.csv(drug_data_encoded, "drug_preprocessed.csv", row.names = FALSE)
