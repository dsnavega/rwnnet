# This file is part of rwnnet
#
# Copyright (C) 2021, David Senhora Navega
#
# rwnnet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# rwnnet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with rwnnet. If not, see <http:#www.gnu.org/licenses/>.
#
# David Senhora Navega
# Laboratory of Forensic Anthropology
# Department of Life Sciences
# University of Coimbra
# Calçada Martim de Freitas, 3000-456, Coimbra
# Portugal

#' rwnnet: An R/C++ implementation of randomized weights neural networks.
#'
#' rwnnet implements randomized feed-forward neural network where the hidden
#' layers are random initialized and fixed during training. The network is
#' fitted in a single-pass by training a regularized least squares read-out
#' layer - Tikhonov regularization through an efficient SVD-based algorithm.
#' As an additional regularization mechanism gaussian noise is added both to
#' the inputs and output(s) of the network.
#' The key components of are written in C++, using Rcpp, and binded to R for
#' maximum performance and ease-of-use. Shallow and deep network can be
#' trained using the Extreme Learning Machine and the Random Vector Functional
#' Link frameworks.  Supports regression, multi-output regression and
#' classification. Depending on the algorithm chosen it allows to fit an
#' implicit ensemble model.
#'
#' @author David Senhora Navega
#'
#' @export
#'
#' @param x a data.frame or tibble of numeric inputs.
#' @param y a numeric vector (regression), a factor (classification) or a
#' data.frame (tibble) of numeric outputs (multi-output regression).
#' @param size a vector of integers defining the width and depth of network, i.e.
#' c(32, 32) defines two layers with 32 nodes (neurons) each.
#' @param algorithm a character with the type of algorithm used to train the
#' network. See Details.
#' @param skip a logical. If TRUE (default) skip connections are added. See
#' Details.
#' @param flat a logical. If TRUE (default) when possible the structure of the
#' network is flattened - in a deep or multi-layer network - and trained as a
#' shallow network by using skip or direct connections from each hidden layer
#' to the output layer.
#' @param eta a numeric value. Controls the gaussian noise regularization.
#' See Details
#'
rwnnet <- function(x, y, size, algorithm, skip = TRUE, flat = TRUE, eta = 1) {

  # Start Stopwatch
  start <- Sys.time()

  # Data Validation ----
  task <- n_group <- group_names <- NULL

  if (any(is.na(x)))
    stop("\n(-) NA values not allowed in x.")

  if (any(is.na(y)))
    stop("\n(-) NA values not allowed in y.")


  if (NROW(x) != NROW(y))
    stop("\n(-) Number of instances in x and y do not match.")

  if (isFALSE(inherits(x, what = "data.frame")))
    stop("\n(-) x must be a data.frame or tibble.")

  if (isFALSE(all(sapply(x, is.numeric))))
    stop("\n(-) All columns of x must be numeric.")

  if (isTRUE(inherits(y, what = "data.frame"))) {

    if (isFALSE(all(sapply(y, is.numeric))))
      stop("\n(-) If y is a tibble or data.frame, columns must be numeric.")

    task <- "regression"

  } else {

    if (is.numeric(y))
      task <- "regression"

    if (is.factor(y)) {

      task <- "classification"

      n_group <- nlevels(y)
      group_names <- levels(y)

      y <- 2 * factor_to_matrix(y) - 1

    }

  }

  x <- data.matrix(x)
  y <- data.matrix(y)

  # Argument Validation ----

  # size
  if (missing(size)) {
    size <- NULL
  } else {

    if (isFALSE(is.vector(size)))
      stop("\n(-) size must be a vector")

  }

  # algorithm
  if (missing(algorithm))
    algorithm = NULL

  algorithms = c(
    "slrwnnet",   # Shallow Randomized Network (ELM or RVFL)
    "drwnnet",    # Deep+ Fully Randomized Network (RVFL)
    "edrwnnet",   # Ensemble Deep Fully Randomized Network (RVFL)
    "aerwnnet",   # Stacked Auto-encoding Network (Shallow/Deep+)
    "edaerwnnet", # Ensemble Stacked Auto-encoding Network (Deep*)
    "saerwnnet",  # Supervised Auto-encoding Network (Shallow/Deep*)
    "r2rwnnet"    # Recursive Randomized Network (Deep*)
  )

  algorithm = match.arg(arg = algorithm, choices = algorithms)

  # skip
  if (missing(skip))
    skip <- TRUE

  if (!is.logical(skip))
    stop("\n(-) skip must be a logical.")

  # flat
  if (missing(flat))
    flat <- TRUE

  if (!is.logical(flat))
    stop("\n(-) flat must be a logical.")

  # eta
  if (missing(eta))
    eta <- 1

  if (!is.numeric(eta))
    stop("\n(-) eta must be a numeric value.")

  if (eta < 0.0 | eta > 2.575829) {
    stop("\n(-) eta must be a value between 0.0 and 2.575829.")
  }

  # Training ----

  # Size Heuristic
  if (algorithm == "slrwnnet") {

    if (is.null(size)) {
      k <- floor(log2(nrow(x)))
      size <- floor((8 * sqrt((2 ^ k) / k)))
      size <- 2 ^ ceiling(log2(size))
    }

  } else {

    if (is.null(size)) {
      k <- floor(log2(nrow(x)))
      ksize <- floor((8 * sqrt((2 ^ k) / k)))
      dsize <- c(2 * ksize, ksize, ksize / 2)
      size <- 2 ^ floor(log2(dsize))
    }

  }

  # Training Algorithms
  network <- switch (algorithm,

    slrwnnet = {
      .fit_slrwnnet(
        x = x, y = y, size = size[1], skip = skip, eta = eta
      )
    },

    drwnnet = {
      .fit_drwnnet(
        x = x, y = y, size = size, eta = eta
      )
    },

    edrwnnet = {
      .fit_edrwnnet(
        x = x, y = y, size = size, eta = eta
      )
    },

    aerwnnet = {
      .fit_aerwnnet(
        x = x, y = y, size = size, skip = skip, flat = flat, eta = eta
      )
    },

    edaerwnnet = {
      .fit_edaerwnnet(
        x = x, y = y, size = size, skip = skip, eta = eta
      )
    },

    saerwnnet = {
      .fit_saerwnnet(
        x = x, y = y, size = size, eta = eta
      )
    },

    r2rwnnet = {
      .fit_r2rwnnet(
        x = x, y = y, size = size, skip = skip, eta = eta
      )
    }

  )

  # Class Object ----

  object <- structure(

    .Data = list(
      network = network,
      size = if (algorithm == "slrwnnet") max(size) else size,
      algorithm = algorithm,
      skip = skip,
      flat = flat,
      eta = eta,
      task = task,
      n_group = n_group,
      group_names = group_names,
      n_cases = nrow(x),
      n_input = ncol(x),
      input_names = colnames(x),
      time = difftime(Sys.time(), start, units = "secs")[[1]]
    ),

    class = "rwnnet"

  )

  invisible(object)

}

#' Predict Method for rwnnet
#'
#' @param object a fitted object of class inherithng from "rwnnet".
#' @param x  data.frame or tibble in which to look for inputs with which to
#' predict. If omitted, the leave-one-out cross-validation results are used.
#' @param ... ...
#'
#' @export
#'
predict.rwnnet <- function(object, x, ...) {

  if (isFALSE(inherits(object, what = "rwnnet")))
    stop("\n(-) 'rwnnet' object required.")

  if (missing(x)) {

    # Leave-One-Out Cross-Validation
    predicted <- object$network[["loocv"]]

  } else {

    # Data Validation ----

    if (any(is.na(x)))
      stop("\n(-) NA values not allowed in x.")

    if (isFALSE(inherits(x, what = "data.frame")))
      stop("\n(-) x must be a data.frame or tibble.")

    if (isFALSE(sapply(x, is.numeric)))
      stop("\n(-) All columns of x must be numeric.")

    if (all(object$input_names %in% colnames(x))) {

      x <- x[, object$input_names, drop = FALSE]

    } else {

      stop("\n(-) Not all inputs used to fit network are present on x.")

    }

    x <- data.matrix(x)

    # Output ----

    predicted <- switch (object$algorithm,

      slrwnnet = {
        .predict_slrwnnet(network = object$network, x = x)
      },

      drwnnet = {
        .predict_drwnnet(
          network = object$network, x = x
        )
      },

      edrwnnet = {
        .predict_edrwnnet(
          network = object$network, x = x
        )
      },

      aerwnnet = {
        .predict_aerwnnet(
          network = object$network, x = x
        )
      },

      edaerwnnet = {
        .predict_edaerwnnet(
          network = object$network, x = x
        )
      },

      saerwnnet = {
        .predict_saerwnnet(
          network = object$network, x = x
        )
      },

      r2rwnnet = {
        .predict_r2rwnnet(
          network = object$network, x = x
        )
      }

    )

  }

  # Convert to score to probability via sparsemax
  if (object$task == "classification") {

    predicted <- predicted - apply(predicted, 1, max)
    predicted <- t(apply(predicted, 1, sparsemax))
    colnames(predicted) <- object$group_names

  }

  return(predicted)

}

#' Print Method for rwnnet
#'
#' @export
#' @noRd
#'
print.rwnnet <- function(x, ...) {
  cat("\n - Randomized Neural Network - \n")
  cat("\n Task:", x$task)
  cat("\n Algorithm:", x$algorithm)
  cat("\n Size:", x$size)
  cat("\n Depth:", length(x$size))
}
