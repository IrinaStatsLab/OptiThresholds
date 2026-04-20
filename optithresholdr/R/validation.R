# Normalize public loss names.
.normalize_loss <- function(loss) {
  loss <- tolower(as.character(loss)[1L])
  if (!loss %in% c("loss1", "loss2")) {
    stop("`loss` must be 'loss1' or 'loss2'.", call. = FALSE)
  }
  loss
}

# Normalize public Wasserstein distance names.
.normalize_wdist <- function(wdist) {
  wdist <- toupper(as.character(wdist)[1L])
  if (!wdist %in% c("W1", "W2")) {
    stop("`wdist` must be 'W1' or 'W2'.", call. = FALSE)
  }
  wdist
}

# Validate scalar count arguments such as K and M.
.as_count <- function(x, name, minimum = 0L) {
  if (length(x) != 1L || !is.finite(x) || x != as.integer(x) || x < minimum) {
    stop(sprintf("`%s` must be a single integer >= %d.", name, minimum), call. = FALSE)
  }
  as.integer(x)
}

# Validate an optional measurement range.
.normalize_range <- function(range) {
  if (is.null(range)) {
    return(NULL)
  }

  if (length(range) != 2L || any(!is.finite(range)) || range[1L] >= range[2L]) {
    stop("`range` must be a length-2 numeric vector with range[1] < range[2].", call. = FALSE)
  }

  as.numeric(range)
}

# Resolve a data-frame column from either the default name or a supplied string.
.resolve_column <- function(data, value, default, label) {
  if (is.null(value)) {
    value <- default
  } else if (!is.character(value) || length(value) != 1L || !nzchar(value)) {
    stop(sprintf("`%s` must be NULL or a single string.", label), call. = FALSE)
  }

  if (!value %in% names(data)) {
    stop(sprintf("Could not find the `%s` column.", label), call. = FALSE)
  }

  value
}

# Turn supported inputs into a named list of subject-level vectors.
.subject_list <- function(x, id = NULL, value = NULL) {
  if (is.data.frame(x)) {
    # For data-frame input, default to columns named `id` and `gl` unless the
    # caller explicitly supplies different columns.
    id_col <- .resolve_column(x, id, "id", "id")
    value_col <- .resolve_column(x, value, "gl", "value")
    keep <- !is.na(x[[id_col]]) & !is.na(x[[value_col]])
    return(split(as.numeric(x[[value_col]][keep]), as.character(x[[id_col]][keep])))
  }

  if (!is.list(x)) {
    stop("`x` must be a list of subject vectors, a data frame, or a distribution object.", call. = FALSE)
  }

  labels <- names(x)
  if (is.null(labels)) {
    labels <- paste0("subject_", seq_along(x))
  } else {
    blanks <- !nzchar(labels)
    labels[blanks] <- paste0("subject_", which(blanks))
  }

  names(x) <- labels
  x
}

# Drop missing values, reject non-finite values, and sort each subject vector.
.clean_subject <- function(x) {
  x <- as.numeric(x)
  x <- x[!is.na(x)]

  if (any(!is.finite(x))) {
    stop("Subject measurements must be finite after removing missing values.", call. = FALSE)
  }

  sort(x)
}

# Combine free and fixed cutoffs and enforce the measurement range.
.merge_cutoffs <- function(cutoffs, fixed, range) {
  cutoffs <- as.numeric(cutoffs)
  fixed <- as.numeric(fixed)
  merged <- sort(c(cutoffs, fixed))

  if (any(!is.finite(merged))) {
    stop("Cutoffs must be finite numeric values.", call. = FALSE)
  }

  if (length(merged) && (merged[1L] < range[1L] || merged[length(merged)] > range[2L])) {
    stop("Cutoffs must lie within the measurement range.", call. = FALSE)
  }

  merged
}

# Require callers beyond preprocessing to pass an existing distribution object.
.require_distribution <- function(x, name) {
  if (!inherits(x, "optithreshold_distribution")) {
    stop(
      sprintf("`%s` must be an `optithreshold_distribution`. Call `as_distribution()` first.", name),
      call. = FALSE
    )
  }

  x
}
