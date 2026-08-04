## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning = TRUE,
  message = TRUE,
  echo = TRUE,
  fig.width = 7,
  fig.height = 5,
  class.source = "fold-show",
  result = "asis",
  dpi = 100,
  dev = "png"
)

## ----setup-libraries, warning = FALSE, message = FALSE------------------------
library(tidyverse)
library(labelled)
library(ggplot2)
library(splines)
library(lme4)
library(marginaleffects)
library(ADNIMERGE2)

## ----setup-parameters---------------------------------------------------------
theme_set(theme_bw(base_size = 12))
# Color for DX group
dx_color_pal <- c("#73C186", "#F2B974", "#DF957C", "#999999")

## ----consort-diagram, echo = FALSE, fig.width = 8.5, fig.alt = "Analysis Population Diagram"----
# Data prep for consort diagram
consort_data <- ADSL %>%
  filter(ENRLFL %in% "Y") %>%
  select(USUBJID, ENRLFL, DX) %>%
  left_join(
    ADQS %>%
      filter(PARAMCD %in% "ADASTT13") %>%
      group_by(USUBJID) %>%
      summarize(NUM_NON_MISSING = sum(!is.na(AVAL))) %>%
      ungroup(),
    by = "USUBJID"
  ) %>%
  assert_uniq(USUBJID) %>%
  mutate(NUM_NON_MISSING = ifelse(is.na(NUM_NON_MISSING), 0, NUM_NON_MISSING)) %>%
  mutate(
    enrolled = case_when(ENRLFL %in% "Y" ~ "Enrolled"),
    not_bl_dx = case_when(is.na(DX) ~ "Have no a baseline diagnostics summary"),
    bl_dx = factor(
      case_when(!is.na(DX) ~ as.character(DX)),
      levels = c("CN", "MCI", "DEM"),
      labels = c("Cognitive Normal (CN)\n", "Mild Cognitive Impairment (MCI)\n", "Dementia (DEM)")
    ),
    missing_score_value = case_when(
      NUM_NON_MISSING == 0 | is.na(NUM_NON_MISSING) ~ "Have no at least one ADAS-Cog\n Item-13 Score"
    ),
    known_score_value = ifelse(NUM_NON_MISSING != 0, "Analysis Population", NA_character_)
  )

consort::consort_plot(
  data = consort_data,
  orders = c(
    enrolled = "Enrolled Population",
    not_bl_dx = "Excluded",
    bl_dx = "Have a Baseline \n Diagnostics Summary",
    missing_score_value = "Excluded",
    known_score_value = "Analysis Population"
  ),
  side_box = c("not_bl_dx", "missing_score_value"),
  allocation = "bl_dx",
  cex = 0.8
)

## ----data-prep----------------------------------------------------------------
# Prepare analysis dataset of ADAS-cog item-13 score
ADADAS <- ADQS %>%
  # Enrolled participant
  filter(ENRLFL %in% "Y") %>%
  # ADAS-cog item-13 total score
  filter(PARAMCD %in% "ADASTT13") %>%
  # Compute time variable in years
  mutate(
    TIME = convert_number_days(AVISITN, unit = "year"),
    DX = factor(DX, levels = levels(ADSL$DX))
  ) %>%
  # For spline term in the model
  filter(!if_any(all_of(c("TIME", "DX", "AVAL")), ~ is.na(.x)))

## ----check-analysis-popu, echo = FALSE----------------------------------------
# Total analysis population - consort data
num_final_popu_consort <- consort_data %>%
  filter(!is.na(bl_dx) & !is.na(known_score_value)) %>%
  nrow()

# Total analysis population - ADADAS data
num_final_popu_adadas <- ADADAS %>%
  distinct(USUBJID) %>%
  nrow()

if (num_final_popu_adadas != num_final_popu_consort) {
  stop("Number of analysis population is not the same!")
}

## ----indiv-profile-plot, warning = FALSE, fig.alt = "Individual profile plot"----
# Individual profile (spaghetti) plot
individual_profile_plot <- ADADAS %>%
  ggplot(aes(x = TIME, y = AVAL, group = USUBJID, color = DX)) +
  geom_line(alpha = 0.75) +
  scale_x_continuous(breaks = seq(0, max(ADADAS$TIME, na.rm = TRUE), 2)) +
  scale_color_manual(values = dx_color_pal) +
  labs(
    y = "ADAS-cog Item-13 Total Score",
    x = "Timeline since baseline visit (in years)",
    color = "Baseline Diagnostics Status",
    caption = paste0(
      "Based on known baseline diagnostics status ",
      "(i.e., any missing diagnostics status is excluded)"
    ),
    plot.caption = element_text(hjust = 0)
  ) +
  theme(legend.position = "bottom")
individual_profile_plot

## ----spline-term--------------------------------------------------------------
# Spline term defined in the global environment
ns21 <- function(t) {
  as.numeric(predict(splines::ns(ADADAS$TIME,
    df = 2,
    Boundary.knots = c(0, max(ADADAS$TIME))
  ), t)[, 1])
}
ns22 <- function(t) {
  as.numeric(predict(splines::ns(ADADAS$TIME,
    df = 2,
    Boundary.knots = c(0, max(ADADAS$TIME))
  ), t)[, 2])
}
assign("ns21", ns21, envir = .GlobalEnv)
assign("ns22", ns22, envir = .GlobalEnv)

## ----model-fit, warning = TRUE, message = TRUE--------------------------------
# Fit linear mixed effect model using spline terms to account
#  non-linear trend of time effect

# With a random slope term ----
lmer_mod_fit <- lme4::lmer(
  formula = AVAL ~ (I(ns21(TIME)) + I(ns22(TIME))) * DX + (TIME | USUBJID),
  data = ADADAS,
  control = lmerControl(optimizer = "Nelder_Mead")
)

lmer_mod_fit

## ----model-check, fig.alt = "Posterior predictive plot - model assumptions check"----
library(performance)
# To relabel x-axis and getting ggplot2 object, set panel = FALSE
model_check <- performance::check_model(
  x = lmer_mod_fit,
  check = "pp_check",
  panel = FALSE
)
model_check <- plot(model_check)
model_check$PP_CHECK +
  labs(x = "ADAS-cog Item-13 Total Score")

## ----model-predict, fig.width = 7, fig.alt = "Predicted score by baseline diagnostics status"----
# Population level prediction
pred_value <- marginaleffects::predictions(
  model = lmer_mod_fit,
  conf_level = 0.95,
  re.form = NA,
  newdata = expand_grid(
    TIME = seq(0, max(ADADAS$TIME), by = 0.03),
    DX = levels(ADSL$DX),
    USUBJID = NA
  ) %>%
    filter(!(DX %in% "DEM" & TIME > 2))
)

predicted_plot <- pred_value %>%
  ggplot(aes(x = TIME, y = estimate, color = DX)) +
  geom_line() +
  geom_ribbon(
    aes(ymin = conf.low, ymax = conf.high, fill = DX),
    alpha = 0.15, linetype = 0, show.legend = FALSE
  ) +
  scale_x_continuous(breaks = seq(0, max(pred_value$TIME, na.rm = TRUE), 2)) +
  scale_fill_manual(values = dx_color_pal) +
  scale_color_manual(values = dx_color_pal) +
  labs(
    x = "Timeline since baseline visit (in years)",
    y = "Predicted ADAS-cog Item-13 \n Total Score with 95% CI",
    color = "Baseline Diagnostics Status",
    fill = NULL,
    caption = paste0(
      "The prediction timeline for Dementia (DEM) diagnostics status was limited",
      " to 2 years due to the\n anticipated follow-up assessment collection ",
      "in the study."
    )
  ) +
  theme(
    legend.position = "bottom",
    plot.caption = element_text(hjust = 0)
  )
predicted_plot

