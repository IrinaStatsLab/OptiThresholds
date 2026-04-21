# Loss computation helpers and public loss evaluation.

#' Evaluate A Threshold Loss
#'
#' Compute the loss value for a given set of candidate cutoffs. This is the
#' same objective function minimized by `optimal_thresholds()`.
#'
#' @param distribution An `optithreshold_distribution` created by
#'   `as_distribution()`.
#' @param cutoffs Numeric vector of candidate thresholds to evaluate. These
#'   must lie within the measurement `range` of the distribution.
#' @param loss Loss function to use. `"loss1"` (distribution preservation)
#'   measures how well each subject's original distribution is approximated by
#'   its TIR summary. `"loss2"` (distance preservation) measures how well
#'   pairwise subject distances are maintained after thresholding; requires
#'   at least 2 subjects.
#' @param wdist Wasserstein distance order: `"W2"` (default, squared Euclidean
#'   transport cost) or `"W1"` (absolute transport cost, more robust to
#'   subjects with extreme measurements).
#' @param fixed Optional numeric vector of fixed thresholds that are merged
#'   with `cutoffs` before the loss is computed (for semi-supervised
#'   evaluation).
#'
#' @return A scalar loss value.
#' @export
evaluate_loss <- function(distribution, cutoffs, loss = c("loss1", "loss2"),
                          wdist = c("W2", "W1"), fixed = NULL) {
  distribution <- .require_distribution(distribution, "distribution")
  loss <- .normalize_loss(loss)
  wdist <- .normalize_wdist(wdist)

  if (loss == "loss1") {
    return(mean(.loss1_distances(distribution, cutoffs, fixed = fixed, wdist = wdist)))
  }

  if (length(distribution$sorted_data) < 2L) {
    stop("`loss2` requires at least two subject distributions.", call. = FALSE)
  }

  # Loss2 compares the pairwise subject distance structure induced by the
  # original and amalgamated quantile representations.
  q_a <- .cutoff_amalgamation(distribution, cutoffs, fixed = fixed)
  dist_amalg <- .pairwise_upper_distances(q_a, wdist, distribution$denominator)
  mean((.loss2_reference(distribution, wdist) - dist_amalg) ^ 2)
}

# Pairwise subject distances stored in condensed upper-triangle form.
.pairwise_upper_distances <- function(qtiles, wdist, denominator) {
  wdist <- .normalize_wdist(wdist)
  if (nrow(qtiles) < 2L) {
    return(numeric())
  }

  if (wdist == "W2") {
    return(as.vector(stats::dist(qtiles, method = "euclidean")) / sqrt(denominator))
  }

  as.vector(stats::dist(qtiles, method = "manhattan")) / denominator
}

# Loss1 compares each original subject quantile curve to its amalgamated curve.
.loss1_distances <- function(distribution, cutoffs, fixed = NULL, wdist = "W2") {
  q_a <- .cutoff_amalgamation(distribution, cutoffs, fixed = fixed)
  diffs <- distribution$qtiles - q_a

  if (.normalize_wdist(wdist) == "W2") {
    rowSums(diffs * diffs) / distribution$denominator
  } else {
    rowSums(abs(diffs)) / distribution$denominator
  }
}

# Cache the baseline pairwise distances used by Loss2.
.loss2_reference <- function(distribution, wdist = "W2") {
  key <- paste0("loss2_reference_", .normalize_wdist(wdist))
  if (!exists(key, envir = distribution$loss2_reference_cache, inherits = FALSE)) {
    assign(
      key,
      .pairwise_upper_distances(distribution$qtiles, wdist, distribution$denominator),
      envir = distribution$loss2_reference_cache
    )
  }
  get(key, envir = distribution$loss2_reference_cache, inherits = FALSE)
}
