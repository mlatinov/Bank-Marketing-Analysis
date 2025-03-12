
#### Libraries ####
library(tidyverse)
library(corrplot)
library(RColorBrewer)

## Load the data 
bank_full <- read_delim("Data/data_raw/bank-full.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)

# Change the char features into factors
bank_eda <- bank_full %>% mutate(across(where(is.character), as.factor))
bank_eda_log <- bank_full %>% mutate(across(where(is.numeric), log1p))

# Exclude duration because is not known before a call is performed. Also, after the end of the call y is obviously known
bank_eda <- bank_eda %>%
  select(-duration)

bank_eda_log <- bank_eda_log %>%
  select(-duration)

## Function to select only num values and pivot longer the dataset
only_numeric_pivot <- function(data){
  data %>%
    select_if(is.numeric)%>%
    pivot_longer(cols = everything(),names_to = "Features",values_to = "Value")
  }

## Plot the numerical distributions
only_numeric_pivot(bank_eda)%>%
  ggplot(aes(x = Value,fill = after_stat(count)))+
  geom_histogram()+
  scale_fill_viridis_c(option = "magma")+
  facet_wrap(~Features,scale = "free_x")+
  theme_minimal()+
  labs(
    title = " Features Distributions",
    x = "Feature",
    y = "Count")+
  theme(
    title = element_text(size = 14),
    axis.title = element_text(size = 13),
    strip.text = element_text(size = 12)
  )
  
## Boxplot 
only_numeric_pivot(bank_eda)%>%
  ggplot(aes(x = Value))+
  geom_boxplot()+
  facet_wrap(~Features,scale = "free_x")+
  theme_minimal()+
  labs(
    title = "Features Boxplot",
    subtitle = "Strong skewness and a high presence of outliers")+
  theme(
    strip.text = element_text(size = 11),
    title = element_text(size = 13),
    axis.title.x = element_text(size = 14))
  
# Correlation Matrix
bank_eda %>%
  select_if(is.numeric) %>%
  cor() %>%
  corrplot(
    method = "color",    
    order = "hclust",
    addCoef.col = "black",         
    tl.col = "black",              
    tl.srt = 45,                  
    col = brewer.pal(n = 8, name = "PuRd"),  
    cl.pos = "b",                 
    diag = FALSE                  
  )

## 2D Density plot
bank_eda_log%>%
  filter(pdays >=0)%>%
ggplot(aes(x = age, y = balance)) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_viridis_c(option = "magma") + 
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_minimal() +
  labs(
    title = "Density plot Age and Balance",
    x = "Log(age)",
    y = "Log(balance)")+
  theme(
    legend.position = "right",
    title = element_text(size = 14),
    axis.title = element_text(size = 12)
    )  





