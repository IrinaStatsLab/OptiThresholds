test_that("as_distribution stores the expected baseline quantiles", {
  expect_equal(unname(fixture_distribution$qtiles), fixture_qtiles, tolerance = 1e-10)
  expect_equal(fixture_distribution$range, fixture_range, tolerance = 1e-12)
  expect_equal(fixture_distribution$M, fixture_M)
})

test_that("list and data-frame inputs produce equivalent distributions", {
  df <- data.frame(
    id = rep(names(fixture_subjects), lengths(fixture_subjects)),
    gl = unlist(fixture_subjects, use.names = FALSE)
  )
  custom_df <- data.frame(
    person = df$id,
    glucose = df$gl
  )

  from_df <- as_distribution(df, range = fixture_range, M = fixture_M)
  from_string_df <- as_distribution(
    custom_df,
    id = "person",
    value = "glucose",
    range = fixture_range,
    M = fixture_M
  )

  expect_equal(from_df$qtiles, fixture_distribution$qtiles, tolerance = 1e-10)
  expect_equal(from_string_df$qtiles, fixture_distribution$qtiles, tolerance = 1e-10)
})

test_that("data-frame column arguments are string-only", {
  df <- data.frame(
    person = rep(names(fixture_subjects), lengths(fixture_subjects)),
    glucose = unlist(fixture_subjects, use.names = FALSE)
  )

  expect_error(
    as_distribution(df, id = 1, value = "glucose", range = fixture_range, M = fixture_M),
    "`id` must be NULL or a single string"
  )
})

test_that("missing values are dropped before distributions are formed", {
  distribution <- as_distribution(
    list(subject_1 = c(1, NA_real_, 2), subject_2 = c(NA_real_, NA_real_), subject_3 = c(4, 5)),
    range = c(0, 10),
    M = 4
  )

  expect_equal(names(distribution$sorted_data), c("subject_1", "subject_3"))
  expect_equal(distribution$sorted_data[[1L]], c(1, 2))
  expect_equal(distribution$sorted_data[[2L]], c(4, 5))
})

test_that("invalid subject inputs are rejected", {
  expect_error(as_distribution(list(numeric())), "No observed values")
  expect_error(as_distribution(list(c(1, Inf))), "finite")
  expect_error(as_distribution(list(c(NA_real_))), "No observed values")
  expect_error(as_distribution("not valid"), "must be a list")
})
