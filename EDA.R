# Exploratory Data Analysis (EDA)

library(tidyverse)
library(lubridate)
library(ggplot2)
library(DataExplorer)  # For automated EDA report
library(corrplot)
library(GGally)


# 1. Load Data

df <- read.csv("../Dataset/Test/Netflix Userbase.csv", stringsAsFactors = FALSE)

# Convert data types
df$Join_Date <- as.Date(df$Join_Date, format = "%m/%d/%Y")
df$Last_Payment_Date <- as.Date(df$Last_Payment_Date, format = "%m/%d/%Y")
df$Subscription_Type <- as.factor(df$Subscription_Type)
df$Country <- as.factor(df$Country)
df$Gender <- as.factor(df$Gender)
df$Device <- as.factor(df$Device)


# 2. Overview

str(df)
summary(df)
skimr::skim(df)   


# 3. Variable Analysis

# Target variable distribution
ggplot(df, aes(x = Subscription_Type, fill = Subscription_Type)) +
  geom_bar() +
  labs(title = "Distribution of Subscription Types", x = "Subscription Type", y = "Count") +
  theme_minimal()

# Numeric variables
ggplot(df, aes(x = Age)) +
  geom_histogram(bins = 20, fill = "orange") +
  labs(title = "Distribution of Age")

# Categorical variables
ggplot(df, aes(x = Device, fill = Device)) +
  geom_bar() +
  labs(title = "Device Distribution")


# 4. Correlation Analysis

# Revenue by Subscription Type
ggplot(df, aes(x = Subscription_Type, y = Monthly_Revenue, fill = Subscription_Type)) +
  geom_boxplot() +
  labs(title = "Monthly Revenue by Subscription Type")

# Subscription Type by Gender
ggplot(df, aes(x = Subscription_Type, fill = Gender)) +
  geom_bar(position = "dodge") +
  labs(title = "Subscription Type by Gender")

# Subscription Type by Device
ggplot(df, aes(x = Subscription_Type, fill = Device)) +
  geom_bar(position = "dodge") +
  labs(title = "Subscription Type by Device")

# Subscription Type by Country (Top 10 countries only)
top_countries <- df %>%
  count(Country, sort = TRUE) %>%
  top_n(10)

ggplot(df %>% filter(Country %in% top_countries$Country),
       aes(x = Country, fill = Subscription_Type)) +
  geom_bar(position = "dodge") +
  labs(title = "Subscription Type by Top 10 Countries") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 7. EDA HTML report (Optional)

create_report(df, output_file = "EDA_Report.html")
