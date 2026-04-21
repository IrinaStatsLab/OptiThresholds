\name{OptiThresholdR-package}
\alias{OptiThresholdR-package}
\alias{OptiThresholdR}
\title{Data-Driven Thresholds for Distributional Wearable Data}
\description{
The \pkg{OptiThresholdR} package provides preprocessing and optimization tools
for data-driven threshold determination using Wasserstein-based loss functions.
It supports a simple workflow: preprocess repeated wearable measurements with
\code{as_distribution()}, score candidate cutoffs with \code{evaluate_loss()},
and estimate cutoffs for time-in-range summaries with
\code{optimal_thresholds()} so that the resulting summaries preserve
distributional information from the original measurement profiles.
}
\details{
The package is intended for repeated wearable measurements such as continuous
glucose monitor data. It uses the full participant-level measurement profiles
when estimating interpretable thresholds. Missing measurement values are
removed before subject distributions are formed.
}
\docType{package}
