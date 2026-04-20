#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace Rcpp;

namespace {

// Evaluate the sample quantile function of a sorted subject vector.
double interpolate_sorted_prob(const NumericVector& sorted_sample, double prob) {
  const int n = sorted_sample.size();
  if (n == 1) {
    return sorted_sample[0];
  }

  const double index = prob * (n - 1.0) + 1.0;
  const int lower = std::max(1, std::min(n, static_cast<int>(std::floor(index))));
  const int upper = std::max(1, std::min(n, static_cast<int>(std::ceil(index))));
  const double weight = index - lower;

  return sorted_sample[lower - 1] * (1.0 - weight) + sorted_sample[upper - 1] * weight;
}

// Build the unique CDF knots induced by the cutoff set for one subject.
NumericVector build_knots(const NumericVector& sorted_sample, const NumericVector& cutoffs) {
  const int n = sorted_sample.size();
  const int n_cutoffs = cutoffs.size();
  std::vector<double> knots;
  knots.reserve(n_cutoffs + 2);
  knots.push_back(0.0);

  int position = 0;
  for (int i = 0; i < n_cutoffs; ++i) {
    const double cutoff = cutoffs[i];
    while (position < n && sorted_sample[position] <= cutoff) {
      ++position;
    }
    const double knot = static_cast<double>(position) / static_cast<double>(n);
    if (knot != knots.back()) {
      knots.push_back(knot);
    }
  }

  if (knots.back() != 1.0) {
    knots.push_back(1.0);
  }

  return wrap(knots);
}

// Interpolate subject quantiles from the knot locations onto the shared grid.
NumericVector interpolate_grid(const NumericVector& knots, const NumericVector& values, const NumericVector& grid) {
  const int m = grid.size();
  NumericVector out(m);
  const int k = knots.size();
  int interval = 0;

  for (int i = 0; i < m; ++i) {
    const double x = grid[i];
    if (x <= knots[0]) {
      out[i] = values[0];
      continue;
    }
    if (x >= knots[k - 1]) {
      out[i] = values[k - 1];
      continue;
    }

    while (interval + 1 < k && knots[interval + 1] < x) {
      ++interval;
    }

    const double x0 = knots[interval];
    const double x1 = knots[interval + 1];
    const double y0 = values[interval];
    const double y1 = values[interval + 1];
    const double weight = (x - x0) / (x1 - x0);
    out[i] = y0 + weight * (y1 - y0);
  }

  return out;
}

}  // namespace

// Amalgamate each subject distribution onto the common quantile grid.
// [[Rcpp::export]]
NumericMatrix cutoff_amalgamation_cpp_impl(List sorted_data, NumericVector cutoffs, NumericVector grid) {
  const int n_subjects = sorted_data.size();
  const int grid_size = grid.size();
  NumericMatrix out(n_subjects, grid_size);

  for (int i = 0; i < n_subjects; ++i) {
    NumericVector sorted_sample = sorted_data[i];
    NumericVector knots = build_knots(sorted_sample, cutoffs);
    const int knot_size = knots.size();
    NumericVector q_knots(knot_size);

    for (int j = 0; j < knot_size; ++j) {
      q_knots[j] = interpolate_sorted_prob(sorted_sample, knots[j]);
    }

    out(i, _) = interpolate_grid(knots, q_knots, grid);
  }

  return out;
}
