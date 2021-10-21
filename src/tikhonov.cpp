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

#ifndef TIKHONOV_H
#define TIKHONOV_H

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

//[[Rcpp::export(.tikhonov)]]
Rcpp::List tikhonov(arma::mat x, arma::mat y) {

  // Regularization Search Space
  vec lambda = regspace(-6.0, 2.0, 12.0);
  int delta = lambda.n_elem;
  for (int i = 0; i < delta; i++){
    lambda(i) = pow(2.0, lambda(i));
  }

  arma::vec loss = arma::ones(delta);

  // Declare SVD
  arma::mat U, V;
  arma::vec S;

  if (x.n_rows > x.n_cols) {

    // Compute SVD
    arma::svd_econ(U, S, V, x);

    // Efficient LOOCV (Optimization)
    for(int i = 0; i < delta; i++) {
      arma::vec theta = arma::square(S) / (arma::square(S) + lambda(i));
      arma::mat gamma = U.t();
      gamma.each_col() %= theta;
      arma::mat H = arma::sum(U % gamma.t(), 1);
      arma::mat Z = U * gamma * y;
      arma::mat E = y - Z;
      E.each_col() /= (1.0 - H);
      loss(i) = arma::accu(arma::square(E)) / x.n_rows;
    };

    // Coefficients (Beta)
    arma::vec theta = S / (arma::square(S) + lambda(loss.index_min()));
    arma::mat rho = arma::ones(size(U.t()));
    rho.each_col() %= theta;
    arma::mat beta =  V * (rho % U.t() * y);

    // Efficient LOOCV (Predictions)
    theta = arma::square(S) / (arma::square(S) + lambda(loss.index_min()));
    arma::mat gamma = U.t();
    gamma.each_col() %= theta;
    arma::mat H = arma::sum(U % gamma.t(), 1);
    y.each_col() %= H;
    arma::mat O = ((x * beta) - y);
    O.each_col() /= (1.0 - H);

    return Rcpp::List::create(
      Rcpp::Named("weights") = beta,
      Rcpp::Named("loocv") = O,
      Rcpp::Named("lambda") = lambda(loss.index_min())
    );

  } else {

    // Compute SVD
    arma::svd_econ(U, S, V, x.t());

    // Efficient LOOCV (Optimization)
    for(int i = 0; i < delta; i++) {
      arma::vec theta = arma::square(S) / (arma::square(S) + lambda(i));
      arma::mat gamma = V.t();
      gamma.each_col() %= theta;
      arma::mat H = arma::sum(V % gamma.t(), 1);
      arma::mat Z = V * gamma * y;
      arma::mat E = y - Z;
      E.each_col() /= (1.0 - H);
      loss(i) = arma::accu(arma::square(E)) / x.n_rows;
    };

    // Coefficients (Beta)
    arma::vec theta = S / (arma::square(S) + lambda(loss.index_min()));
    arma::mat rho = arma::ones(size(V.t()));
    rho.each_col() %= theta;
    arma::mat beta =  U * (rho % V.t() * y);

    // Efficient LOOCV (Predictions)
    theta = arma::square(S) / (arma::square(S) + lambda(loss.index_min()));
    arma::mat gamma = V.t();
    gamma.each_col() %= theta;
    arma::mat H = arma::sum(V % gamma.t(), 1);
    y.each_col() %= H;
    arma::mat O = (x * beta) - y;
    O.each_col() /= (1.0 - H);

    return Rcpp::List::create(
      Rcpp::Named("weights") = beta,
      Rcpp::Named("loocv") = O,
      Rcpp::Named("lambda") = lambda(loss.index_min())
    );

  }

}

#endif
