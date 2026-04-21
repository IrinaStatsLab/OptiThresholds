#' Create A Distribution Object
#'
#' Preprocess subject-level measurements into the distribution object used by
#' the loss and optimization routines. Each subject's measurements are
#' converted to an empirical quantile function on a shared grid, which is the
#' internal representation for all downstream computations.
#'
#' @param x A list of numeric vectors (one per subject), a long-format data
#'   frame, or an existing `optithreshold_distribution` (returned as-is).
#' @param id Column name (string) for subject IDs when `x` is a data frame.
#'   Defaults to `"id"` if omitted.
#' @param value Column name (string) for measurement values when `x` is a
#'   data frame. Defaults to `"gl"` if omitted.
#' @param range Length-2 numeric vector giving the physical measurement range
#'   of the device, e.g., `c(40, 400)` for a CGM recording 40--400 mg/dL. If
#'   `NULL`, the observed data range is used with a small buffer. All
#'   measurements must fall within this range.
#' @param M Number of interior quantile grid points used to discretize each
#'   subject's quantile function (the stored grid has `M + 2` points including
#'   endpoints). The default `M = 200` gives a good balance of accuracy and
#'   speed; larger values improve the Wasserstein distance approximation at
#'   modest extra cost.
#'
#' @return An S3 `optithreshold_distribution` object with sorted subject
#'   measurements and quantiles on a shared grid. It also stores a small
#'   internal cache for the baseline `loss2` distances so repeated objective
#'   evaluations do not recompute the same reference distances.
#' @export
as_distribution <- function(x, id = NULL, value = NULL, range = NULL, M = 200) {
  if (inherits(x, "optithreshold_distribution")) {
    return(x)
  }
  
  # Validation
  M <- .as_count(M, "M", minimum = 1L)
  range <- .normalize_range(range)

  # Clean and sort each subject once so later loss evaluations can reuse the
  # same cached representation.
  sorted_data <- lapply(
    .subject_list(x, id = id, value = value),
    .clean_subject
  )
  sorted_data <- Filter(length, sorted_data)

  if (!length(sorted_data)) {
    stop("No observed values remain after removing missing values.", call. = FALSE)
  }

  labels <- names(sorted_data)
  sample_sizes <- lengths(sorted_data)

  data_min <- min(vapply(sorted_data, `[`, numeric(1), 1L))
  data_max <- max(vapply(sorted_data, function(subject) subject[length(subject)], numeric(1)))

  if (is.null(range)) {
    range <- c(data_min - 1e-8, data_max + 1e-8)
  } else if (data_min < range[1L] || data_max > range[2L]) {
    stop("Data are outside the supplied measurement range.", call. = FALSE)
  }

  # Store subject quantiles on a fixed grid so both losses can work with matrix
  # operations instead of rebuilding quantiles at every evaluation.
  grid <- seq(0, 1, length.out = M + 2L)
  qtiles <- t(vapply(sorted_data, .interpolate_sorted_sample, numeric(M + 2L), probs = grid))
  rownames(qtiles) <- labels

  structure(
    list(
      sorted_data = sorted_data,
      sample_sizes = sample_sizes,
      range = as.numeric(range),
      M = M,
      denominator = M + 1L,
      grid = grid,
      qtiles = qtiles,
      loss2_reference_cache = new.env(parent = emptyenv())
    ),
    class = "optithreshold_distribution"
  )
}

# Evaluate the sample quantile function of a pre-sorted subject vector.
.interpolate_sorted_sample <- function(sorted_sample, probs) {
  n <- length(sorted_sample)
  if (n == 1L) {
    return(rep(sorted_sample, length(probs)))
  }

  index <- probs * (n - 1) + 1
  lower <- pmax(1L, pmin(n, floor(index)))
  upper <- pmax(1L, pmin(n, ceiling(index)))
  weight <- index - lower

  sorted_sample[lower] * (1 - weight) + sorted_sample[upper] * weight
}

# Merge free and fixed cutoffs, then run the compiled amalgamation routine.
.cutoff_amalgamation <- function(distribution, cutoffs, fixed = NULL) {
  cutoffs <- .merge_cutoffs(cutoffs, fixed, distribution$range)
  cutoff_amalgamation_cpp_impl(distribution$sorted_data, cutoffs, distribution$grid)
}

#' @export
print.optithreshold_distribution <- function(x, ...) {
  cat("<optithreshold_distribution>\n")
  cat("Subjects:", length(x$sorted_data), "\n")
  cat("Range:", paste(format(x$range, trim = TRUE), collapse = " to "), "\n")
  cat("Quantile grid points:", x$M + 2L, "\n")
  invisible(x)
}
