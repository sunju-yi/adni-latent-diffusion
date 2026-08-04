library(testthat)
library(tidyverse)

# PET Scan Tracer Type in ADNI4 ----
if (exists("AMYREAD")) {
  test_that("Check ADNI4 PET scan tracer type", {
    pre_specified_tracer_type <- c(
      "18F-Florbetaben/NeuraCeqTM (Threshold SUVR 1.12)",
      "18F-AV-45/Florbetapir/AmyvidTM (Threshold SUVR 1.17)",
      "18F-NAV4694 (Threshold SUVR 1.14)"
    )
    reported_tracer_type <- AMYREAD %>%
      filter(COLPROT == "ADNI4") %>%
      mutate(TRACERTYPE = as.character(TRACERTYPE)) %>%
      distinct(TRACERTYPE) %>%
      na.omit() %>%
      pull()

    expect_equal(
      object = reported_tracer_type,
      expected = pre_specified_tracer_type,
      ignore_attr = TRUE,
      check.attributes = FALSE,
      check.names = FALSE,
      tolerance = 1.5e-8,
      info = "Check ADNI4 PET scan tracer type"
    )
  })
}
