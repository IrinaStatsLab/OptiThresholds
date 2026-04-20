test_that("K = 0 returns the fixed thresholds without optimization", {
  fit <- optimal_thresholds(
    fixture_distribution,
    K = 0,
    loss = "loss1",
    fixed = c(1.5, 7.0)
  )

  expect_equal(fit$cutoffs, c(1.5, 7.0))
  expect_equal(
    fit$objective,
    evaluate_loss(fixture_distribution, numeric(), loss = "loss1", fixed = c(1.5, 7.0)),
    tolerance = 1e-10
  )
})

test_that("optimal_thresholds returns sorted thresholds and a consistent objective", {
  fit <- optimal_thresholds(
    fixture_distribution,
    K = 2,
    loss = "loss1",
    seed = 1,
    control = list(NP = 20, itermax = 10, trace = FALSE)
  )

  internal <- fit$cutoffs
  expect_true(all(diff(internal) >= 0))
  expect_equal(
    fit$objective,
    evaluate_loss(fixture_distribution, internal, loss = "loss1"),
    tolerance = 1e-8
  )
})

test_that("optimal_thresholds keeps fixed thresholds in the final result", {
  fit <- optimal_thresholds(
    fixture_distribution,
    K = 1,
    loss = "loss2",
    wdist = "W1",
    fixed = 1.5,
    seed = 1,
    control = list(NP = 10, itermax = 8, trace = FALSE)
  )

  expect_true(any(abs(fit$cutoffs - 1.5) < 1e-12))
  expect_true(all(diff(fit$cutoffs) >= 0))
  expect_equal(
    fit$objective,
    evaluate_loss(fixture_distribution, fit$cutoffs, loss = "loss2", wdist = "W1"),
    tolerance = 1e-8
  )
})

test_that("initialpop must have one column per free threshold", {
  expect_error(
    optimal_thresholds(
      fixture_distribution,
      K = 1,
      control = list(initialpop = matrix(runif(6), ncol = 2))
    ),
    "one column per free threshold"
  )
})

test_that("optimal_thresholds requires a distribution object", {
  expect_error(
    optimal_thresholds(fixture_subjects, K = 1),
    "Call `as_distribution\\(\\)` first"
  )
})

test_that("fit payload stays lightweight", {
  fit <- optimal_thresholds(
    fixture_distribution,
    K = 1,
    loss = "loss1",
    seed = 1,
    control = list(NP = 10, itermax = 5, trace = FALSE)
  )

  expect_setequal(names(fit), c("cutoffs", "objective", "loss", "wdist", "elapsed"))
})
