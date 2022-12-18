#setwd("~/OneDrive/Documents/4th Year CWRU/CSDS 391 Intro to AI/P2 Files")
library(dplyr)
library(ggplot2)
iris_data <- data.frame(read.csv("irisdata.csv"))

# exercise 1
ex1_df <- select(iris_data, petal_length, petal_width, species)

ggplot(data = ex1_df, aes(x = petal_length, y = petal_width)) +
  geom_point(aes(color = species, 
                 shape = species)) +
  labs(title = "Iris Data: 
       \nRelationship between Petal Length and Width by Species",
       x = "Petal Length (cm)", y = "Petal Width (cm)")



