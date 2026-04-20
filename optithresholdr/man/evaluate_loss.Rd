\name{evaluate_loss}
\alias{evaluate_loss}
\title{Evaluate a Threshold Loss}
\usage{
evaluate_loss(distribution, cutoffs, loss = c("loss1", "loss2"),
  wdist = c("W2", "W1"), fixed = NULL)
}
\arguments{
  \item{distribution}{An \code{optithreshold_distribution}.}
  \item{cutoffs}{Numeric vector of free cutoffs to evaluate.}
  \item{loss}{Loss function, \code{"loss1"} or \code{"loss2"}. \code{loss1}
  averages subject-level discrepancies, while \code{loss2} compares pairwise
  subject distance structure before and after amalgamation.}
  \item{wdist}{Wasserstein base distance, \code{"W2"} or \code{"W1"}.}
  \item{fixed}{Optional fixed thresholds merged with \code{cutoffs} before the
  loss is computed.}
}
\value{
A scalar loss value.
}
\description{
Evaluates the same threshold loss definitions used by the package optimizer.
}
\details{
Call \code{as_distribution()} first when starting from raw subject vectors or a
data frame.
}
\examples{
subjects <- list(
  subject_1 = c(90, 110, 130, 150),
  subject_2 = c(80, 95, 105, 140),
  subject_3 = c(70, 85, 100, 120)
)

distribution <- as_distribution(subjects, range = c(40, 250), M = 25)

evaluate_loss(distribution, cutoffs = c(95, 130), loss = "loss1", wdist = "W2")
evaluate_loss(distribution, cutoffs = c(95, 130), loss = "loss2", wdist = "W1")
}
