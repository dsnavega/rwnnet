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

#include "RcppArmadillo.h"
using namespace arma;
using namespace Rcpp;
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

#include "utilities.h"

arma::mat init_uniform(int n, int m) {

  // Weights & Bias
  arma::mat weights(n + 1, m);
  weights.randu();
  weights *=  2.0;
  weights += -1.0;
  weights /=  2.0;

  return weights;

}

arma::mat init_orthonormal(int n, int m) {

  // Weights & Bias
  arma::mat weights(n + 1, m);
  weights.randn();
  weights = arma::orth(weights);

  return weights;

}

arma::mat init_nguyen_widrow(int n, int m) {

  // Nguyen-Widrow Coefficient
  double beta = 0.7 * pow(n, 1.0 / m);

  // Weights [-0.5, 0.5]
  arma::mat weights(n, m);
  weights.randu();
  weights *=  2.0;
  weights += -1.0;
  weights /= 2.0;

  // Bias [-beta, beta]
  arma::mat bias(1, m);
  bias.randu();
  bias *=  2.0;
  bias += -1.0;
  bias /= 2.0;
  bias *= beta;

  // Normalize and Scale
  weights = beta * arma::normalise(weights);
  weights.insert_rows(0, bias);

  return weights;

}

arma::mat init_tapson(arma::mat x, int m) {

  // Weights, Tapson et al. 2014
  int n = x.n_rows;
  Rcpp::NumericVector ones = Rcpp::NumericVector::create(-1, 1);
  Rcpp::NumericVector r_ij = sample(ones, m * n, true);
  arma::mat R = arma::mat(r_ij.begin(), m, n, false);
  arma::mat weights = R * x;
  weights = arma::normalise(weights.t(), 1, 0);

  // Bias [-0.5, 0.5]
  arma::mat bias(1, m);
  bias.randu();
  bias *=  2.0;
  bias += -1.0;
  bias /= 2.0;

  weights.insert_rows(0, bias);

  return weights;

}

arma::mat initialise_weights(arma::mat x, int size, std::string init) {

  // Number of Inputs
  int n = x.n_cols;

  // Scheme
  if (init == "uniform") {return init_uniform(n, size);}

  else if (init == "orthonormal") {
    return init_orthonormal(n, size);
  }

  else if (init  == "nguyen-widrow") {
    return init_nguyen_widrow(n, size);
  }

  else if (init  == "tapson") {
    return init_tapson(x, size);
  }

  else {
    std::string msg  = "(-) Invalid intialisation scheme: " + init;
    Rcpp::stop(msg);
  }

}
