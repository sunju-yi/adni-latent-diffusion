## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning = FALSE,
  message = FALSE,
  echo = FALSE
)

## ----dataset-list, echo = FALSE-----------------------------------------------
# Libraries
library(tidyverse)
library(ADNIMERGE2)
library(DT)

source(system.file("dataset-list.R", package = "ADNIMERGE2", mustWork = TRUE))

dataset_list <- get_required_dataset_list(
  use_type = "article",
  add_url_link = TRUE
) %>%
  mutate(text = case_when(
    !is.na(script_list) ~ paste0(article_list, ",", script_list),
    TRUE ~ article_list
  )) %>%
  arrange(data_code) %>%
  concat_dataset_url(.data = ., var_name = "data_code") %>%
  select(
    `Dataset Code` = data_code,
    `Dataset Label` = label,
    `Use Articles/Scripts` = article_list,
    `Derived Dataset` = source_derived_data
  ) %>%
  DT::datatable(.,
    options = list(paging = FALSE, searchable = TRUE, bInfo = FALSE),
    escape = FALSE,
    caption = "List of datasets that requwired to replicated ADNIMERGE2 R data package"
  )
dataset_list

