\name{as_distribution}
\alias{as_distribution}
\title{Create a Distribution Object}
\usage{
as_distribution(x, id = NULL, value = NULL, range = NULL, M = 200)
}
\arguments{
  \item{x}{A list of numeric vectors, a data frame, or an existing
  \code{optithreshold_distribution}.}
  \item{id}{Optional subject column when \code{x} is a data frame. If omitted,
  the package uses a column named \code{id}. If supplied, it must be a
  string.}
  \item{value}{Optional measurement column when \code{x} is a data frame. If
  omitted, the package uses a column named \code{gl}. If supplied, it must be a
  string.}
  \item{range}{Optional measurement range. If \code{NULL}, the observed range is
  expanded by a small symmetric buffer.}
  \item{M}{Number of interior quantile grid points. The stored grid has
  \code{M + 2} points because it includes both endpoints.}
}
\value{
An S3 \code{optithreshold_distribution} object with sorted subject
measurements and quantiles on a shared grid. It also stores a small internal
cache for the baseline \code{loss2} distances so repeated objective evaluations
do not recompute the same reference distances.
}
\description{
Preprocess subject-level measurements into the distribution object used by the
loss and optimization routines.
}
\details{
Missing measurement values are removed before subject distributions are formed.
Subjects with no observed values left are dropped.
}
\examples{
subjects <- list(
  subject_1 = c(90, 110, 130, 150),
  subject_2 = c(80, 95, 105, 140)
)

as_distribution(subjects, range = c(40, 250), M = 25)

df <- data.frame(
  person = c("A", "A", "B", "B"),
  glucose = c(90, 110, 80, 140)
)

as_distribution(df, id = "person", value = "glucose", range = c(40, 250), M = 25)
}
