# Create clinical state diagram -----
#' @title Create Clinical State Space Diagram
#' @param output_file Output file path, in `png` image type
#' @return The result `png` image file will be created according to the input file path.
#' @examples 
#' \dontrun{
#' create_state_space(output_file = "./inst/state_space_diagram.png")
#' }
#' @rdname create_state_space
#' @export 
#' @importFrom DiagrammeR create_node_df create_edge_df create_graph set_node_position export_graph
#' @importFrom dplyr mutate case_when

create_state_space <- function(output_file) {
  require(dplyr)
  require(DiagrammeR)

  # Node data
  pal <- c("#64AC59FF", "#E69F00", "#000000", "#FFFFFF")
  ## Clinical state
  clinical_state <- c(
    "Cognitive Normal\n (CN)",
    "Mild Cognitive \n Impairment\n (MCI)",
    "Dementia\n (DEM)",
    "Death\n (F)"
  )
  transition_type <- c("Converter", "Reverter", "Absorber")
  node_data <- DiagrammeR::create_node_df(
    n = length(clinical_state),
    label = clinical_state,
    style = "filled",
    fontsize = 12,
    fontcolor = pal[3],
    fixedsize = FALSE,
    fontname = "Helvetica",
    shape = "rectangle",
    fillcolor = pal[4],
    color = pal[3],
  )

  # Edges data
  # Direction arrows
  edge_label <- paste0("q@_{", paste0(c("cm", "cd", "cf", "md", "mf", "mc", "df", "dm", "dc"), "}"))
  edge_data <- DiagrammeR::create_edge_df(
    from = c(rep(1, 3), rep(2, 3), rep(3, 2)),
    to = c(2:4, 3:4, 1, 4, 2),
    label = edge_label,
    rel = "leading_to",
    arrowhead = "normal",
    arrowtail = "icurve",
    arrowsize = 1,
    style = "solid",
    fontsize = 12
  ) %>%
    dplyr::mutate(
      style = dplyr::case_when(
        to == 4 ~ "dotted",
        TRUE ~ style
      ),
      color = case_when(
        from > to & to != 4 ~ pal[1],
        from <= to & to != 4 ~ pal[2],
        to == 4 ~ pal[3]
      )
    )

  # Diagram/flowchart graph
  graph <- DiagrammeR::create_graph(
    nodes_df = node_data,
    edges_df = edge_data,
    directed = TRUE,
    attr_theme = "fdp"
  ) %>%
    DiagrammeR::set_node_position(node = 1, x = 5, y = 20) %>%
    DiagrammeR::set_node_position(node = 2, x = 9, y = 20.75) %>%
    DiagrammeR::set_node_position(node = 3, x = 13, y = 20) %>%
    DiagrammeR::set_node_position(node = 4, x = 9, y = 17)

  DiagrammeR::export_graph(
    graph = graph,
    file_name = output_file,
    file_type = "PNG", 
    width = 1000, 
    height = 700
  )
  #invisible(output_file)
  output_graph <- DiagrammeR::render_graph(graph = graph, as_svg = TRUE)
  return(output_graph)
}


# Utilities function for `msm` model fit -----
#' @title Get `msm.fit` Summary for Multiple Time Points
#' @param msm_fit `msm_fit` objects
#' @param type Type of summary either `pmatrix.msm`, `ppass.msm` or `totlos.msm`, Default: 'pmatrix.msm'
#' @param time Numeric vector of time points,  Default: 1
#' @inherit msm pmatrix.msm
#' @inherit msm totlos.msm
#' @inherit msm ppass.msm
#' @return A data.frame
#' @seealso
#'  \code{\link[msm]{pmatrix.msm}}, \code{\link[msm]{ppass.msm}}, \code{\link[msm]{totlos.msm}}
#'  \code{\link[broom]{reexports}}
#' @rdname get.msm.fit.summary
#' @export
#' @importFrom msm pmatrix.msm ppass.msm totlos.msm
#' @importFrom broom tidy
#' @importFrom dplyr bind_rows
#' @importFrom tibble as_tibble

get.msm.fit.summary <- function(msm_fit, type = "pmatrix.msm",
                                t = 1, ci = "normal", cl = 0.95, B = 1000,
                                exclude_state_num = NULL, ...) {
  time <- NULL
  check_summary_type(type)
  if (type %in% summary_type()[1]) summary_fun <- function(...) msm::pmatrix.msm(...)
  if (type %in% summary_type()[2]) summary_fun <- function(...) msm::ppass.msm(...)
  if (type %in% summary_type()[3]) summary_fun <- function(...) msm::totlos.msm(...)

  long_data <- lapply(t, function(t1) {
    summary_fun(x = msm_fit, t = t1, ci = ci, cl = cl, B = B, ...) %>%
      broom::tidy() %>%
      dplyr::mutate(time = as.numeric(t1))
  }) %>%
    dplyr::bind_rows() %>%
    tibble::as_tibble()

  if (!is.null(exclude_state_num)) {
    long_data <- exclude_absorber_state(
      .data = long_data,
      state_num = exclude_state_num,
      type = type
    )
  }
  return(long_data)
}

#' @title Exclude for Absorber State
#' @description FUNCTION_DESCRIPTION
#' @param .data A data.frame
#' @param state_num Absosrber state number, Default: NA
#' @inheritParams get_summary.msm.fit
#' @return A data.frame that does not includes records of the specified absorber state
#' @rdname exclude_absorber_state
#' @export
exclude_absorber_state <- function(.data, state_num = NA, type) {
  check_summary_type(type)
  col_name <- c("state", "tostate")
  if (type %in% summary_type()[3]) {
    col_name <- col_name[2]
  }
  check_colnames(.data = .data, col_names = col_name, strict = TRUE)

  .data <- .data %>%
    filter(!if_all(any_of(col_name), ~ .x %in% state_num))

  return(.data)
}

#' @title Check `msm` Summary Type
#' @inheritParams get_summary.msm.fit
#' @return An error message if the provided type is not existed.
check_summary_type <- function(type) {
  rlang::arg_match0(arg = type, values = summary_type())
}

#' @title `msm` Summary Type
#' @return A character string of `pmatrix.msm` and `ppass.msm`.
summary_type <- function() {
  return(c("pmatrix.msm", "ppass.msm", "totlos.msm"))
}


#' @title Convert `msm.prevalence` result into long format
#' @param .data `msm.prevalence` object
#' @return A long format data.frame that contains `time`, `state` and `estimate` variables.
#' @seealso 
#'  \code{\link[tibble]{rownames}}
#' @rdname convert_to_longformat
#' @export 
#' @importFrom tibble rownames_to_column
#' @importFrom dplyr pivot_longer mutate
#' @importFrom tidyselect everything
convert_to_longformat <- function(.data) {
  
  # if (!"msm.prevalence" %in% class(prev.msm)) {
  #   stop("The input object must be a `msm.prevalence` object.")
  # }
  .data <- .data %>%
    as.data.frame() %>%
    tibble::rownames_to_column(var = "time") %>%
    pivot_longer(
      cols = -time,
      names_to = "state",
      values_to = "estimate"
    ) %>%
    mutate(across(everything() & -estimate, as.character))
  
  return(.data)
}

# Prepare the prevalence value
prepare_prevalence <- function(prev.msm, type = "num") {
  require(tidyverse)
  rlang::arg_match0(arg = type, values = c("num", "prop"))
  if (type == "num") {
    observ_list_name <- "Observed"
    expected_list_name <- "Expected"
  } else {
    observ_list_name <- "Observed percentages"
    expected_list_name <- "Expected percentages"
  }
  
  observed_data <- prev.msm %>%
    pluck(., observ_list_name) %>%
    convert_to_longformat() %>%
    mutate(type = "Observed")
  
  # Expected values
  expected_data <- prev.msm %>%
    pluck(., expected_list_name)
  # Estimate
  expected_data_estimate <- expected_data %>%
    pluck(., "estimates") %>%
    convert_to_longformat()
  
  if ("ci" %in% names(expected_data)) {
    # Confidence interval
    expected_data_ci <- expected_data %>%
      pluck(., "ci") %>%
      convert_to_longformat() %>%
      rename("state_temp" = state) %>%
      mutate(
        state = as.character(str_extract(state_temp, "^.{1}")),
        ci_type = as.numeric(str_remove_all(state_temp, "^.{2}|\\%")),
        ci_type = case_when(
          ci_type <= 50 ~ "lower",
          ci_type > 50 ~ "upper"
        )
      ) %>%
      select(-state_temp) %>%
      pivot_wider(
        id_cols = everything(),
        names_from = ci_type,
        values_from = estimate
      ) %>%
      select(-time, -state)
    
    if (nrow(expected_data_estimate) != nrow(expected_data_ci)) {
      stop("Check row numbers for expected data")
    }
    expected_data <- bind_cols(expected_data_estimate, expected_data_ci)
  } else {
    expected_data <- expected_data_estimate
  }
  
  expected_data <- expected_data %>%
    mutate(type = "Estimated")
  
  output_data <- observed_data %>%
    bind_rows(expected_data) %>%
    mutate(
      state = str_remove_all(state, "State "),
      type = factor(type, levels = c("Observed", "Estimated"))
    ) %>%
    mutate(across(c(time, estimate), as.numeric))
  
  return(output_data)
}
