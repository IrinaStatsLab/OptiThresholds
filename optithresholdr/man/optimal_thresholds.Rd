\name{optimal_thresholds}
\alias{optimal_thresholds}
\title{Optimize Data-Driven Thresholds}
\usage{
optimal_thresholds(x, K, loss = c("loss1", "loss2"),
  wdist = c("W2", "W1"), fixed = NULL, seed = NULL,
  control = list())
}
\arguments{
  \item{x}{An \code{optithreshold_distribution}.}
  \item{K}{Number of free thresholds to optimize.}
  \item{loss}{Loss function, \code{"loss1"} or \code{"loss2"}.}
  \item{wdist}{Wasserstein base distance, \code{"W2"} or \code{"W1"}.}
  \item{fixed}{Optional fixed thresholds that remain in the final threshold
  set.}
  \item{seed}{Optional random seed for reproducibility. If supplied,
  \code{set.seed(seed)} is called immediately before optimization.}
  \item{control}{A named list passed through to
  \code{DEoptim::DEoptim.control()}. Package defaults are
  \code{NP = max(4, 15 * K)}, \code{itermax = 1000}, \code{CR = 0.7},
  \code{F = 0.75}, \code{strategy = 3}, \code{reltol = 1e-4},
  \code{steptol = 30}, \code{bs = FALSE}, \code{trace = FALSE}, and
  \code{parallelType = "none"}. If \code{initialpop} is supplied, its row
  count becomes the effective \code{NP}.}
}
\value{
An S3 \code{optithreshold_fit} object containing the optimized cutoffs,
the achieved objective value, the selected loss and Wasserstein distance, and
runtime.
}
\description{
Runs a \pkg{DEoptim}-based threshold search with sorted-candidate
canonicalization.
}
\details{
Call \code{as_distribution()} first when starting from raw subject vectors or a
data frame.

The solver still canonicalizes each candidate threshold vector into sorted
order before evaluation, but the control layer is intentionally lightweight and
otherwise delegates argument handling to \pkg{DEoptim}.
}
\examples{
\donttest{
subjects <- list(
  subject_1 = c(90, 110, 130, 150, 180),
  subject_2 = c(75, 82, 95, 105, 125, 160),
  subject_3 = c(60, 70, 88, 92, 115, 140)
)

distribution <- as_distribution(subjects, range = c(40, 250), M = 25)

fit <- optimal_thresholds(
  distribution,
  K = 2,
  loss = "loss1",
  seed = 1,
  control = list(NP = 20, itermax = 50, trace = FALSE)
)

fit
}
}
