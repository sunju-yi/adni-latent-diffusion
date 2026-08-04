## ----setup, include = FALSE---------------------------------------------------
library(knitr)
knitr::opts_chunk$set(
  collapse = TRUE,
  fig.width = 5,
  fig.height = 4,
  comment = "#>",
  class.source = "fold-show"
)
# Package Versions
pkgName <- "ADNIMERGE2"
pkgFilename <- paste0("ADNIMERGE2_", packageVersion("ADNIMERGE2"), ".tar.gz")

## ----study-pkg----------------------------------------------------------------
library(ADNIMERGE2)

## ----data-date----------------------------------------------------------------
# Data source downloaded date
ADNIMERGE2::DATA_DOWNLOADED_DATE

## ----raw-data-dict------------------------------------------------------------
# Data dictionary for raw data
head(ADNIMERGE2::DATADIC, 6)

## ----derived-data-dict--------------------------------------------------------
# Data dictionary for derived data
head(ADNIMERGE2::DERIVED_DATADIC, 6)

## ----derived-data-dict2-------------------------------------------------------
# Data dictionary for derived data based on R6-class object
ADNIMERGE2::METACORES

## ----libraries, echo=TRUE, message=FALSE, warning=FALSE-----------------------
library(tidyverse)

## ----codes-all----------------------------------------------------------------
# Get variable code values for all available data based on the DATADIC
data_dict_codes <- get_factor_levels_datadict(
  .datadic = ADNIMERGE2::DATADIC,
  tbl_name = NULL,
  nested_value = FALSE
)

class(data_dict_codes)

data_dict_codes %>%
  datadict_as_tibble() %>%
  relocate(prefix, suffix) %>%
  head()

## ----replace-missing-values-4-------------------------------------------------
# Convert "-4" into missing value
convert_to_missing_value(
  .data = ADNIMERGE2::DXSUM,
  col_name = colnames(ADNIMERGE2::DXSUM),
  value = "-4",
  missing_char = NA,
  phase = adni_phase()
) %>%
  select(-PTID) %>%
  head(., 0)

## ----replace-missing-values-1-------------------------------------------------
# Convert "-1" into missing value in ADNI1 phase only
convert_to_missing_value(
  .data = ADNIMERGE2::DXSUM,
  col_name = NULL,
  value = "-1",
  missing_char = NA,
  phase = "ADNI1"
) %>%
  select(-PTID) %>%
  head(., 0)

