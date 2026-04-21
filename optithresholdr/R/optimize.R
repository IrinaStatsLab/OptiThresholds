# Optimization wrappers and result formatting.

#' Optimize Data-Driven Thresholds
#'
#' Find `K` thresholds that minimize the chosen loss function over the
#' measurement range, using differential evolution (DE). The returned
#' cutoffs define `K + 1` bins for time-in-range summaries.
#'
#' @param x An `optithreshold_distribution` created by `as_distribution()`.
#' @param K Number of free (optimized) thresholds. Together with the range
#'   endpoints, `K` thresholds define `K + 1` time-in-range bins. For example,
#'   `K = 4` produces the same granularity as the standard CGM consensus
#'   thresholds (54, 70, 181, 251 mg/dL).
#' @param loss Loss function to minimize. `"loss1"` (distribution
#'   preservation) is best for summarizing a single cohort; `"loss2"` (distance
#'   preservation) is best for mixed-population data where between-group
#'   separation matters. See `evaluate_loss()` for details.
#' @param wdist Wasserstein distance order: `"W2"` (default) or `"W1"`. `W1`
#'   is more robust to subjects with extreme measurements.
#' @param fixed Optional numeric vector of fixed thresholds that are kept in
#'   the final set (semi-supervised mode). For example, `fixed = c(70, 181)`
#'   keeps the standard CGM range while optimizing `K` additional cutoffs
#'   around it. `K` refers only to the number of *free* thresholds.
#' @param seed Optional random seed for reproducibility. If supplied,
#'   `set.seed(seed)` is called immediately before optimization.
#' @param control A named list passed to `DEoptim::DEoptim.control()`. Package
#'   defaults are `NP = max(4, 15 * K)`, `itermax = 1000`, `CR = 0.7`,
#'   `F = 0.75`, `strategy = 3`, `reltol = 1e-4`, `steptol = 30`,
#'   `bs = FALSE`, `trace = FALSE`, and `parallelType = "none"`. Reduce
#'   `itermax` for quick exploration. If `initialpop` is supplied, its row
#'   count becomes the effective `NP`.
#'
#' @return An S3 `optithreshold_fit` object with components:
#'   \describe{
#'     \item{cutoffs}{Numeric vector of all thresholds (free + fixed), sorted.}
#'     \item{objective}{The achieved loss value (lower is better).}
#'     \item{loss}{The loss function used.}
#'     \item{wdist}{The Wasserstein distance used.}
#'     \item{elapsed}{Elapsed optimization time in seconds.}
#'   }
#' @export
optimal_thresholds <- function(x, K, loss = c("loss1", "loss2"),
                               wdist = c("W2", "W1"), fixed = NULL,
                               seed = NULL, control = list()) {
  K <- .as_count(K, "K", minimum = 0L)
  loss <- .normalize_loss(loss)
  wdist <- .normalize_wdist(wdist)
  distribution <- .require_distribution(x, "x")
  fixed <- .merge_cutoffs(numeric(), fixed, distribution$range)
  if (!length(fixed)) {
    fixed <- NULL
  }

  if (loss == "loss2" && length(distribution$sorted_data) < 2L) {
    stop("`loss2` requires at least two subject distributions.", call. = FALSE)
  }

  lower <- rep(distribution$range[1L], K)
  upper <- rep(distribution$range[2L], K)

  # When there are no free thresholds, the result is determined entirely by the
  # fixed thresholds and the boundary endpoints.
  if (K == 0L) {
    objective <- evaluate_loss(distribution, numeric(), loss = loss, wdist = wdist, fixed = fixed)
    merged <- .merge_cutoffs(numeric(), fixed, distribution$range)
    return(.new_fit(
      cutoffs = merged,
      objective = objective,
      loss = loss,
      wdist = wdist,
      elapsed = 0
    ))
  }

  if (!is.null(seed)) {
    if (length(seed) != 1L || !is.finite(seed)) {
      stop("`seed` must be NULL or a single finite number.", call. = FALSE)
    }
    set.seed(as.integer(seed)[1L])
  }

  # DEoptim searches over free thresholds only. Candidates are sorted before
  # evaluation so every point is interpreted as an ordered cutoff vector.
  de_control <- .build_deoptim_control(control, K)
  fn_map <- if (K > 1L) .canonicalize_population else NULL
  objective_fn <- function(par) {
    evaluate_loss(
      distribution,
      .canonicalize_cutoffs(par),
      loss = loss,
      wdist = wdist,
      fixed = fixed
    )
  }

  result <- NULL
  elapsed <- system.time({
    result <- DEoptim::DEoptim(
      fn = objective_fn,
      lower = lower,
      upper = upper,
      control = de_control,
      fnMap = fn_map
    )
  })[["elapsed"]]

  merged <- .merge_cutoffs(result$optim$bestmem, fixed, distribution$range)
  .new_fit(
    cutoffs = merged,
    objective = as.numeric(result$optim$bestval),
    loss = loss,
    wdist = wdist,
    elapsed = elapsed
  )
}

#' @export
print.optithreshold_fit <- function(x, ...) {
  cat("<optithreshold_fit>\n")
  cat("Cutoffs:", if (length(x$cutoffs)) {
    paste(format(signif(x$cutoffs, 6), trim = TRUE), collapse = ", ")
  } else {
    "<none>"
  }, "\n")
  cat("Objective:", format(signif(x$objective, 6), trim = TRUE), "\n")
  cat("Loss:", x$loss, "\n")
  cat("Elapsed seconds:", format(signif(x$elapsed, 6), trim = TRUE), "\n")
  invisible(x)
}

# Build the lightweight result object returned by optimal_thresholds().
.new_fit <- function(cutoffs, objective, loss, wdist, elapsed) {
  fit <- list(
    cutoffs = as.numeric(cutoffs),
    objective = as.numeric(objective),
    loss = loss,
    wdist = wdist,
    elapsed = as.numeric(elapsed)
  )
  class(fit) <- "optithreshold_fit"
  fit
}

# Sort a single cutoff vector into canonical order.
.canonicalize_cutoffs <- function(cutoffs) {
  sort(as.numeric(cutoffs))
}

# Sort each row of a DEoptim population matrix.
.canonicalize_population <- function(pop) {
  if (is.null(dim(pop))) {
    return(sort(as.numeric(pop)))
  }

  if (ncol(pop) == 1L) {
    return(matrix(as.numeric(pop), ncol = 1L))
  }

  t(apply(pop, 1L, sort))
}

# Build the DEoptim control object from package defaults and user overrides.
.build_deoptim_control <- function(control, K) {
  control <- as.list(control)
  if (length(control)) {
    names_control <- names(control)
    if (is.null(names_control) || any(is.na(names_control) | !nzchar(names_control))) {
      stop("`control` must be a named list.", call. = FALSE)
    }
  }

  defaults <- list(
    NP = max(4L, 15L * K),
    itermax = 1000L,
    CR = 0.7,
    F = 0.75,
    strategy = 3L,
    reltol = 1e-4,
    steptol = 30L,
    bs = FALSE,
    trace = FALSE,
    parallelType = "none"
  )

  control <- utils::modifyList(defaults, control)
  control$NP <- max(4L, as.integer(control$NP[1L]))
  control$itermax <- as.integer(control$itermax[1L])
  control$strategy <- as.integer(control$strategy[1L])
  control$bs <- isTRUE(control$bs)

  if (!is.null(control$initialpop)) {
    control$initialpop <- as.matrix(control$initialpop)
    if (ncol(control$initialpop) != K) {
      stop("`control$initialpop` must have one column per free threshold.", call. = FALSE)
    }
    control$NP <- nrow(control$initialpop)
  }

  do.call(DEoptim::DEoptim.control, control)
}
