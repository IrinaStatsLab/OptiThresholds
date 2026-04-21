fixture_subjects <- list(
  subject_1 = c(1, 2, 3, 4),
  subject_2 = c(2, 2, 5, 8),
  subject_3 = c(0, 1, 1, 10)
)

fixture_range <- c(0, 10)
fixture_M <- 4

fixture_distribution <- as_distribution(
  fixture_subjects,
  range = fixture_range,
  M = fixture_M
)

fixture_qtiles <- matrix(
  c(
    1.0, 1.6, 2.2, 2.8, 3.4, 4.0,
    2.0, 2.0, 2.6, 4.4, 6.2, 8.0,
    0.0, 0.6, 1.0, 1.0, 4.6, 10.0
  ),
  nrow = 3,
  byrow = TRUE
)

fixture_cutoffs <- c(2.5, 6.0)
fixture_amalg <- matrix(
  c(
    1.0, 1.6, 2.2, 2.8, 3.4, 4.0,
    2.0, 2.6, 3.2, 4.4, 6.2, 8.0,
    0.0, 0.8666666667, 1.7333333333, 2.6, 4.6, 10.0
  ),
  nrow = 3,
  byrow = TRUE
)

fixture_fixed_cutoffs <- 7.0
fixture_fixed <- 1.5
fixture_amalg_fixed <- matrix(
  c(
    1.0, 1.6, 2.2, 2.8, 3.4, 4.0,
    2.0, 3.0, 4.0, 5.0, 6.2, 8.0,
    0.0, 0.8666666667, 1.7333333333, 2.6, 4.6, 10.0
  ),
  nrow = 3,
  byrow = TRUE
)

fixture_losses <- list(
  loss1_w2 = 0.2592592592592593,
  loss1_w1 = 0.25333333333333347,
  loss2_w2 = 0.054677090355346715,
  loss2_w1 = 0.1354666666666666,
  loss1_w2_fixed = 0.4325925925925924,
  loss1_w1_fixed = 0.37333333333333346,
  loss2_w2_fixed = 0.03718880643103715,
  loss2_w1_fixed = 0.21226666666666652
)

fixture_cutoff_amalgamation <- getFromNamespace(".cutoff_amalgamation", "OptiThresholdR")
