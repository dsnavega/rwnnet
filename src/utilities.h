// This file is part of rwnnet
//
// Copyright (C) 2021, David Senhora Navega
//
// rwnnet is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// rwnnet is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rwnnet. If not, see <http://www.gnu.org/licenses/>.
//
// David Senhora Navega
// Laboratory of Forensic Anthropology
// Department of Life Sciences
// University of Coimbra
// Calçada Martim de Freitas, 3000-456, Coimbra
// Portugal

#ifndef UTILITIES_H
#define UTILITIES_H

#include "RcppArmadillo.h"
using namespace arma;
using namespace Rcpp;
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

Rcpp::List fit_zscore_scaler(arma::mat x);
arma::mat transform_zscore_scaler(List object, arma::mat x);
arma::mat revert_zscore_scaler(List object, arma::mat z);
arma::mat gaussian_noise(arma::mat x, double eta = 1.96);
arma::mat sparse_projection(int n, int m);

arma::mat cov2cor(arma::mat R);
arma::mat whitening_matrix(arma::mat S, std::string method = "PCA");

arma::mat jaccard(arma::mat x, arma::mat z);
arma::mat center_kernel(arma::mat x, arma::mat z);

#endif
