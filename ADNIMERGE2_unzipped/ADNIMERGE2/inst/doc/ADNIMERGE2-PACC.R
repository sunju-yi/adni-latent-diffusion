## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning = FALSE,
  message = FALSE,
  echo = TRUE,
  eval = TRUE
)
# Data print function
datatable <- function(data, paging = FALSE, searchable = TRUE, bInfo = FALSE, ...) {
  DT::datatable(
    data = data, ...,
    options = list(paging = paging, searchable = searchable, bInfo = bInfo, ...)
  )
}

## ----setup-pkgs---------------------------------------------------------------
library(tidyverse)
library(assertr)
library(DT)
library(labelled)

## ----pacc-data-dict, eval = TRUE, include = FALSE, echo = FALSE---------------
# PACC - data dictionary
pacc_data_dic <- bind_rows(
  c(
    FLDNAME = "ORIGPROT",
    LABEL = "Original Study Protocol",
    TYPE = "character",
    TEXT = "First time participation in ADNI study phase"
  ),
  c(
    FLDNAME = "COLPROT",
    LABEL = "Study Data Collection Protocol",
    TYPE = "character"
  ),
  c(
    FLDNAME = "RID",
    LABEL = "Subject Identifier",
    TYPE = "numeric"
  ),
  c(
    FLDNAME = "VISCODE",
    LABEL = "Study visit code",
    TYPE = "character"
  ),
  c(
    FLDNAME = "ADASQ4SCORE",
    LABEL = "Delayed Word Recall Score - portion of ADAS-Cog",
    TYPE = "numeric"
  ),
  c(
    FLDNAME = "ADASQ4SCORE_VISDATE",
    LABEL = "ADAS-Cog Collection/Visit Date",
    TYPE = "date"
  ),
  c(
    FLDNAME = "MMSE",
    LABEL = "Mini-Mental Examination Total Score",
    TYPE = "numeric"
  ),
  c(
    FLDNAME = "MMSE_VISDATE",
    LABEL = "MMSE Collection/Visit Date",
    TYPE = "date"
  ),
  c(
    FLDNAME = "LDELTOTL",
    LABEL = "Logical Memory - Delayed RecallScore",
    TYPE = "numeric"
  ),
  c(
    FLDNAME = "LDELTOTL_VISDATE",
    LABEL = "LDELTOTL Collection/Visit Date",
    TYPE = "date"
  ),
  c(
    FLDNAME = "DIGITSCR",
    LABEL = "Digit Symbol Substitution Score",
    TYPE = "numeric"
  ),
  c(
    FLDNAME = "DIGITSCR_VISDATE",
    LABEL = "DIGITSCR Collection/Visit Date",
    TYPE = "date"
  ),
  c(
    FLDNAME = "TRABSCOR",
    LABEL = "Trial B score",
    TYPE = "numeric"
  ),
  c(
    FLDNAME = "LOG.TRABSCOR",
    LABEL = "Trial B score - Log Scale",
    TYPE = "numeric"
  ),
  c(
    FLDNAME = "TRABSCOR_VISDATE",
    LABEL = "TRABSCOR Collection/Visit Date",
    TYPE = "date"
  ),
  c(
    FLDNAME = "PACC_VISDATE",
    LABEL = "PACC Collection/Visit Date",
    TYPE = "date",
    TEXT = paste0(
      "Either common date across non-missing PACC component ",
      "collection date, the maximum relative date or the corresponding ",
      "study visit date when all component collection date is not available."
    )
  ),
  c(
    FLDNAME = "ADASQ4SCOREZ",
    LABEL = "ADAS Q4 Standardized Score",
    TYPE = "numeric",
    TEXT = paste0(
      "Standardized by corresponding baseline score among ",
      "cognitive normal (CN) new enrollee in ADNI study. ",
      "Also adjusted for directional effect (i.e. multiplied by -1)"
    )
  ),
  c(
    FLDNAME = "MMSEZ",
    LABEL = "Mini-Mental State Examiniation Standardized Score",
    TYPE = "numeric",
    TEXT = paste0(
      "Standardized by corresponding baseline score among ",
      "cognitive normal (CN) new enrollee in ADNI study."
    )
  ),
  c(
    FLDNAME = "LDELTOTLZ",
    LABEL = "Logical Memory - Delayed Recall Standardized Score",
    TYPE = "numeric",
    TEXT = paste0(
      "Standardized by corresponding baseline score among ",
      "cognitive normal (CN) new enrollee in ADNI study."
    )
  ),
  c(
    FLDNAME = "DIGITSCRZ",
    LABEL = "Digit Symbol Substitution Standardized Score",
    TYPE = "numeric",
    TEXT = paste0(
      "Standardized by corresponding baseline score among ",
      "cognitive normal (CN) new enrollee in ADNI1 study phase."
    )
  ),
  c(
    FLDNAME = "LOG.TRABSCORZ",
    LABEL = "Logical Memory - Delayed Recall Log Standardized Score",
    TYPE = "numeric",
    TEXT = paste0(
      "Standardized by corresponding log transformed baseline score among ",
      "cognitive normal (CN) new enrollee in ADNI study. ",
      "Also adjusted for directional effect (i.e. multiplied by -1)"
    )
  ),
  c(
    FLDNAME = "mPACCdigit",
    LABEL = "Modified PACC Digit Score",
    TYPE = "numeric"
  ),
  c(
    FLDNAME = "mPACCtrailsB",
    LABEL = "Modified PACC Trials B Score",
    TYPE = "numeric"
  ),
  c(
    FLDNAME = "ENRLDT",
    LABEL = "Enrollment Date",
    TYPE = "date"
  ),
  c(
    FLDNAME = "ENRLFG",
    LABEL = "Enrollment Flag",
    TYPE = "date"
  ),
  c(
    FLDNAME = "DX",
    LABEL = "Baseline Diagnostics Status",
    TYPE = "character",
    TEXT = paste0(
      "Based on DXSUM eCRF and require to use enrollment flag. Please see ",
      "`get_adni_blscreen_dxsum` function for more information."
    )
  )
) %>%
  tibble::as_tibble() %>%
  relocate(c(FLDNAME, LABEL, TEXT)) %>%
  mutate(
    TEXT = replace_na(TEXT, " "),
    DERIVED = TRUE,
    TBLNAME = "PACC",
    CRFNAME = "[ Derived ] Modified Preclinical Alzheimer’s Cognitive Composite (PACC) Score"
  )

names(pacc_data_dic$LABEL) <- pacc_data_dic$FLDNAME

## ----adjust-scbl--------------------------------------------------------------
# Adjustment of screening-baseline record
id_vars <- c("COLPROT", "RID", "VISCODE", "SCORE_SOURCE")

## MMSE ----
pacc_mmse_long <- pacc_mmse_long %>%
  bind_rows(
    adjust_scbl_record(
      .data = pacc_mmse_long,
      adjust_date_col = "VISDATE",
      check_col = "SCORE"
    ) %>%
      filter(VISCODE %in% get_baseline_vistcode())
  ) %>%
  assert_uniq(all_of(id_vars)) %>%
  set_as_tibble()

## NEUROBAT ----
neurobat_vars <- c("LDELTOTL", "DIGITSCR", "TRABSCOR")

adjusted_bl_neurobat <- adjust_scbl_record(
  .data = pacc_neurobat_long,
  wide_format = FALSE,
  extra_id_cols = "SCORE_SOURCE",
  adjust_date_col = "VISDATE",
  check_col = "SCORE"
) %>%
  # Only baseline records
  filter(VISCODE %in% get_baseline_vistcode())

pacc_neurobat_long <- pacc_neurobat_long %>%
  filter(!VISCODE %in% get_baseline_vistcode()) %>%
  bind_rows(adjusted_bl_neurobat) %>%
  distinct() %>%
  assert_uniq(all_of(id_vars)) %>%
  set_as_tibble()

# ADAS-cog Q4 Score ----
pacc_adas_q4score_long <- pacc_adas_q4score_long %>%
  set_as_tibble()

## ----bind-pacc-input----------------------------------------------------------
# Bind PACC input raw-data
pacc_input_data_long <- bind_rows(
  pacc_mmse_long,
  pacc_neurobat_long,
  pacc_adas_q4score_long
)

# Convert into wide format
pacc_component <- c("ADASQ4SCORE", "MMSE", neurobat_vars)

pacc_input_data_wide <- pacc_input_data_long %>%
  unite(col = "SCORE_VISDATE", SCORE, VISDATE, sep = "-/") %>%
  pivot_wider(
    names_from = SCORE_SOURCE,
    values_from = SCORE_VISDATE
  ) %>%
  assert_uniq(all_of(c("COLPROT", "RID", "VISCODE"))) %>%
  mutate(across(all_of(pacc_component), ~ strsplit(as.character(.), "-/"))) %>%
  unnest_wider(all_of(pacc_component), names_sep = "_") %>%
  rename_with(~ str_remove_all(.x, "\\_1$"), starts_with(pacc_component)) %>%
  rename_with(~ str_replace_all(.x, "2$", "VISDATE"), starts_with(pacc_component)) %>%
  mutate(across(starts_with(pacc_component), ~ ifelse(.x %in% "NA", NA, as.character(.x)))) %>%
  mutate(across(all_of(pacc_component), as.numeric))

## ----pacc-visit-date----------------------------------------------------------
# Add visit date
pacc_input_data_wide <- pacc_input_data_wide %>%
  left_join(
    REGISTRY %>%
      select(COLPROT, RID, VISCODE, EXAMDATE) %>%
      set_as_tibble(),
    by = c("COLPROT", "RID", "VISCODE")
  ) %>%
  verify(nrow(.) == nrow(pacc_input_data_wide))

## ----pacc-common-date---------------------------------------------------------
# Get common date per record
pacc_input_data_wide <- get_vars_common_date(
  .data = pacc_input_data_wide,
  date_cols = paste0(pacc_component, "_VISDATE"),
  compared_ref_date = TRUE,
  ref_date_col = "EXAMDATE",
  select_method = "max",
  preferred_date_col = "COMMON_DATE"
) %>%
  rename("PACC_VISDATE" = FINAL_DATE)

## ----pacc-add-bldx-enrflg-----------------------------------------------------
pacc_input_data_wide <- pacc_input_data_wide %>%
  # Add enrollment date
  left_join(
    get_adni_enrollment(.registry = REGISTRY) %>%
      mutate(RID = as.character(RID)) %>%
      select(ORIGPROT, RID, ENRLDT = EXAMDATE, ENRLFG),
    by = "RID"
  ) %>%
  # Add baseline diagnostics status
  left_join(
    get_adni_blscreen_dxsum(
      .dxsum = DXSUM,
      phase = "Overall",
      visit_type = "baseline"
    ) %>%
      mutate(RID = as.character(RID)) %>%
      select(RID, DX = DIAGNOSIS, DXDATE = EXAMDATE) %>%
      assert_uniq(RID),
    by = "RID"
  )

## ----pacc-component-bl-summary------------------------------------------------
pacc_component_bl <- pacc_input_data_wide %>%
  mutate(TRABSCOR = log(TRABSCOR + 1)) %>%
  # Baseline visit for new enrollee
  filter(ENRLFG %in% "Yes") %>%
  group_by(RID) %>%
  filter(ORIGPROT == COLPROT) %>%
  filter(VISCODE %in% get_baseline_vistcode()) %>%
  ungroup() %>%
  assert_uniq(RID)

pacc_component_bl.summary <- compute_score_summary(
  .data = pacc_component_bl,
  wideFormat = TRUE,
  scoreVar = pacc_component,
  groupVar = "DX",
  filterGroup = "CN"
)

pacc_component_bl.summary %>%
  mutate(VAR = ifelse(VAR == "TRABSCOR", "LOG.TRABSCOR", VAR)) %>%
  rename(`Baseline DX` = DX, `Component` = VAR) %>%
  datatable(
    caption = "PACC Component Standardization Baseline Summary"
  )

## ----compute-pacc-score-------------------------------------------------------
PACC <- compute_pacc_score(
  .data = pacc_input_data_wide,
  bl.summary = pacc_component_bl.summary,
  componentVars = pacc_component,
  rescale_trialsB = TRUE,
  keepComponents = TRUE,
  wideFormat = TRUE
) %>%
  mutate(RID = as.numeric(RID)) %>%
  rename_with(~ str_replace_all(.x, "\\.zscore", "Z"), ends_with(".zscore")) %>%
  create_orig_protocol()

PACC <- PACC %>%
  select(all_of(pacc_data_dic$FLDNAME)) %>%
  mutate(across(all_of(pacc_component), as.numeric)) %>%
  mutate(across(ends_with("_VISDATE"), ~ as.Date(.x))) %>%
  labelled::set_variable_labels(
    .labels = pacc_data_dic$LABEL,
    .strict = TRUE
  )

## ----pacc-data-dic-print, echo = FALSE----------------------------------------
pacc_data_dic %>%
  mutate(TYPE = toupper(TYPE)) %>%
  select(FLDNAME, LABEL, TYPE, TEXT) %>%
  datatable(., paging = TRUE)

## ----bl-pacc------------------------------------------------------------------
# Modified PACC using DSS score
baseline_mPACCdigit <- PACC %>%
  # Only in ADNI1 study phase
  filter(COLPROT == adni_phase()[1]) %>%
  filter(ORIGPROT == COLPROT) %>%
  filter(VISCODE %in% get_baseline_vistcode()) %>%
  # Enrolled subjects
  filter(ENRLFG %in% "Yes") %>%
  assert_uniq(RID) %>%
  select(RID, ORIGPROT, mPACCdigit, DX)

# Modified PACC using Trials B score as a proxy for DSS
baseline_mPACCtrailsB <- PACC %>%
  filter(ORIGPROT == COLPROT) %>%
  filter(VISCODE %in% get_baseline_vistcode()) %>%
  # Enrolled subjects
  filter(ENRLFG %in% "Yes") %>%
  assert_uniq(RID) %>%
  select(RID, ORIGPROT, mPACCtrailsB, DX)

## ----plot-bl-pacc, fig.width = 7, fig.height = 3, fig.alt = "Baseline Modiffied PACC score"----
baseline_PACC <- baseline_mPACCtrailsB %>%
  mutate(
    SCORE = mPACCtrailsB,
    SCORE_NAME = "mPACCtrailsB"
  ) %>%
  bind_rows(
    baseline_mPACCdigit %>%
      mutate(
        SCORE = mPACCdigit,
        SCORE_NAME = "mPACCdigit"
      )
  ) %>%
  group_by(SCORE_NAME) %>%
  mutate(SCORE_NAME = paste0(SCORE_NAME, " (n = ", sum(!is.na(SCORE)), ")")) %>%
  ungroup()

baseline_PACC %>%
  ggplot2::ggplot(aes(x = SCORE)) +
  geom_histogram() +
  facet_wrap(~SCORE_NAME, scales = "free_y") +
  labs(x = "Baseline Modified PACC Score", y = "Count") +
  theme_bw(base_size = 12)

