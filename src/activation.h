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

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "RcppArmadillo.h"
using namespace arma;
using namespace Rcpp;
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

// Rectifiers
arma::mat relu(arma::mat x);
arma::mat softplus(arma::mat x);
arma::mat gelu(arma::mat x);
arma::mat swish(arma::mat x);
arma::mat binary(arma::mat x);

// Transformation (Dot Product) and Activation Helpers
arma::mat transform(arma::mat x, arma::mat w, bool bias = true);
arma::mat activate(arma::mat x, std::string node = "linear");

#endif
