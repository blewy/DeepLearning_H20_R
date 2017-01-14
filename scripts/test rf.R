rm(list = ls())
options(scipen=999)
library(dplyr)
library(stringr)
library(randomForest)
set.seed(88)

train <- read.csv("../input/train.csv", stringsAsFactors = F)
test <- read.csv("../input/test.csv", stringsAsFactors = F)
sub <- read.csv("../input/sample_submission.csv", stringsAsFactors = F)
all_species <- names(sub)[2:100]

# Build a classifier for each species vs. not
for (s in all_species) {
  print(s)
  train$ground_truth <- str_detect(train$species, s)
  rf <- randomForest(as.factor(ground_truth) ~ .,
                     data = train %>% select(-species),
                     num.trees = 300
  )
  yhat <- predict(rf, test, type = "prob")[,2]
  sub[, eval(s)] <- yhat
}

write.csv(sub, "random_forest_benchmark.csv", row.names = F, quote = F)
