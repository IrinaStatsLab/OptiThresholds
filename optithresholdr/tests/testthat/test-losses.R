test_that("cutoff amalgamation matches the stored fixture", {
  expect_equal(
    fixture_cutoff_amalgamation(fixture_distribution, fixture_cutoffs),
    fixture_amalg,
    tolerance = 1e-10
  )

  expect_equal(
    fixture_cutoff_amalgamation(fixture_distribution, fixture_fixed_cutoffs, fixed = fixture_fixed),
    fixture_amalg_fixed,
    tolerance = 1e-10
  )
})

test_that("evaluate_loss matches fixture truth values", {
  expect_equal(
    evaluate_loss(fixture_distribution, fixture_cutoffs, loss = "loss1", wdist = "W2"),
    fixture_losses$loss1_w2,
    tolerance = 1e-10
  )
  expect_equal(
    evaluate_loss(fixture_distribution, fixture_cutoffs, loss = "loss1", wdist = "W1"),
    fixture_losses$loss1_w1,
    tolerance = 1e-10
  )
  expect_equal(
    evaluate_loss(fixture_distribution, fixture_cutoffs, loss = "loss2", wdist = "W2"),
    fixture_losses$loss2_w2,
    tolerance = 1e-10
  )
  expect_equal(
    evaluate_loss(fixture_distribution, fixture_cutoffs, loss = "loss2", wdist = "W1"),
    fixture_losses$loss2_w1,
    tolerance = 1e-10
  )
})

test_that("evaluate_loss handles fixed thresholds", {
  expect_equal(
    evaluate_loss(fixture_distribution, fixture_fixed_cutoffs, loss = "loss1", wdist = "W2", fixed = fixture_fixed),
    fixture_losses$loss1_w2_fixed,
    tolerance = 1e-10
  )
  expect_equal(
    evaluate_loss(fixture_distribution, fixture_fixed_cutoffs, loss = "loss1", wdist = "W1", fixed = fixture_fixed),
    fixture_losses$loss1_w1_fixed,
    tolerance = 1e-10
  )
  expect_equal(
    evaluate_loss(fixture_distribution, fixture_fixed_cutoffs, loss = "loss2", wdist = "W2", fixed = fixture_fixed),
    fixture_losses$loss2_w2_fixed,
    tolerance = 1e-10
  )
  expect_equal(
    evaluate_loss(fixture_distribution, fixture_fixed_cutoffs, loss = "loss2", wdist = "W1", fixed = fixture_fixed),
    fixture_losses$loss2_w1_fixed,
    tolerance = 1e-10
  )
})

test_that("evaluate_loss requires a distribution object", {
  expect_error(
    evaluate_loss(fixture_subjects, fixture_cutoffs, loss = "loss1"),
    "Call `as_distribution\\(\\)` first"
  )
})
