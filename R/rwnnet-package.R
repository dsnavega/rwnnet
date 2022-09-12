# This file is part of rwnnet
#
# Copyright (C) 2022, David Senhora Navega
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

# Dynamic loading and unloading of C++ components ----

#' @useDynLib rwnnet, .registration = TRUE
#' @importFrom Rcpp sourceCpp

.onUnload <- function (libpath) {
  library.dynam.unload("rwnnet", libpath)
}

# Utilities and Tools ----

#' Sparsemax function
#'
#' @author David Senhora Navega
#' @noRd
#'
sparsemax <- function(x) {

  cumsum <- function(x) {
    y <- x
    for (i in 1:length(x)) {
      y[i] <- sum(x[1:i], na.rm = T)
    }
    return(y)
  }

  x_sort <- x[order(x = x, decreasing = T,na.last = T)]
  k <- seq_len(length.out = length(x = x))
  k_array <- 1 + k * x_sort
  x_cumsum <- cumsum(x_sort)
  k_max <-  max(which(k_array > x_cumsum), na.rm = T)
  threshold <- (x_cumsum[k_max] - 1) / k_max
  value <- pmax(x - threshold, 0, na.rm = T)

  return(value)

}

#' Convert factor to indicator matrix
#'
#' @author David Senhora Navega
#' @noRd
#'
factor_to_matrix <- function(x) {

  if (!is.factor(x))
    warning("\n(!) x was coerced to a factor.")

  as_class <- as.factor(x)
  n <- length(as_class)
  m <- nlevels(as_class)
  labeling <- levels(as_class)

  encoding_matrix <- matrix(0, nrow = n, ncol = m)
  index <- (1L:n) + n * (unclass(as_class) - 1L)
  encoding_matrix[index] <- 1
  colnames(encoding_matrix) <- labeling
  rownames(encoding_matrix) <- seq_len(n)

  return(encoding_matrix)

}

#' Convert indicator matrix to factor
#'
#' @author David Senhora Navega
#' @noRd
#'
matrix_to_factor <- function(x) {

  which_maximum <- function(x) {
    if (all(is.na(x))) {
      NA
    } else {
      which.max(x)
    }
  }

  labeling <- colnames(x)
  maximum_value <- apply(x, 1, which_maximum)

  value <- factor(labeling[maximum_value], levels = labeling)
  return(value)

}
