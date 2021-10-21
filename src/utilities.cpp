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

Rcpp::List fit_zscore_scaler(arma::mat x) {

  return Rcpp::List::create(
    Rcpp::Named("mean") = arma::mean(x, 0),
    Rcpp::Named("stddev") = arma::stddev(x, 0),
    Rcpp::Named("k") = x.n_cols
  );

}

arma::mat transform_zscore_scaler(Rcpp::List object, arma::mat x) {

  arma::rowvec mu = object["mean"];
  arma::rowvec sd = object["stddev"];
  int n_features = object["k"];

  for (int i = 0; i < n_features; i++) {

    x.col(i) = x.col(i) - mu(i);
    if (sd(i) != 0.0) {
      x.col(i) = x.col(i) / sd(i);
    }

  }

  return x;

}

arma::mat revert_zscore_scaler(Rcpp::List object, arma::mat z) {

  arma::rowvec mu = object["mean"];
  arma::rowvec sd = object["stddev"];
  int n_features = object["k"];

  for (int i = 0; i < n_features; i++) {

    if (sd(i) != 0.0) {
      z.col(i) = z.col(i) * sd(i);
    }

    z.col(i) = z.col(i) + mu(i);

  }

  return z;

}

arma::mat gaussian_noise(arma::mat x, double eta = 1.96) {

  double noise = eta * (1.0 / sqrt(x.n_rows));
  arma::mat z(x.n_rows, x.n_cols);
  z.randn();
  z *= noise;

  return x + z;

}

arma::mat sparse_projection(int n, int m) {

  double p_one = 1.0 / (2.0  * sqrt(n));
  double p_zero = 1.0 - (1.0 / sqrt(n));
  Rcpp::NumericVector v = Rcpp::NumericVector::create(-1, 0, 1);
  Rcpp::NumericVector p  = Rcpp::NumericVector::create(p_one, p_zero, p_one);
  Rcpp::NumericVector x = sqrt(n) * sample(v, n * m, true, p);
  arma::mat R = (1.0 / sqrt(m)) * arma::mat(x.begin(), n, m, false);
  return R;

}

arma::mat cov2cor(arma::mat R) {

  arma::mat I = sqrt(1.0 / R.diag());
  arma::mat V(R.n_rows, R.n_cols);

  for(uword i = 0; i < R.n_rows; i++) {
    V.col(i).fill(I(i,0));
  }

  R.each_col() %= I;
  R %= V;

  return R;

}

arma::mat whitening_matrix(arma::mat S, std::string method = "PCA") {

  arma::mat W;
  arma::mat eigvec;
  arma::vec eigval;
  arma::uvec index;

  arma::mat variance = S.diag();
  variance = diagmat(1.0 / sqrt(variance));
  arma::mat R = cov2cor(S);

  arma::eig_sym(eigval, eigvec, R, "dc");

  if (method == "PCA") {
    // Fix sign ambiguity by make the diagonal positive
    eigvec *= arma::diagmat(arma::sign(eigvec.diag()));

    W = diagmat(1.0 / sqrt(eigval)) * eigvec.t() * variance;
    W = flipud(W).t();

    variance = arma::sort(eigval,"descencd");
    variance = cumsum(variance/accu(variance));
    index = arma::find(variance < 0.95);

    W = W.cols(index);

  } else if (method == "ZCA") {

    W = eigvec * diagmat(1.0 / sqrt(eigval)) * eigvec.t() * variance;
    W = flipud(W).t();

  }

  return W;

}

arma::mat jaccard(arma::mat x, arma::mat z) {

  arma::mat N(x.n_rows, x.n_cols);
  N.ones();

  arma::mat M(z.n_rows, z.n_cols);
  M.ones();

  arma::mat L(x.n_rows, z.n_rows);
  L = arma::trans((x * M.t()) + (N * z.t()));
  arma::mat J(x.n_rows, z.n_rows);

  // Jaccard Similarity
  J = (z * x.t()) / (L - (z * x.t()));

  // Two vectors of zeros produce nan, remove it.
  J.transform([](double j) {return (std::isnan(j) ? double(1.0) : j);});

  return J;

}

arma::mat center_kernel(arma::mat x, arma::mat z) {

  int n = z.n_rows;
  int m = z.n_cols;

  arma::mat M(m, m);
  M.fill(1.0 / m);

  arma::mat N(n, m);
  N.fill(1.0 / m);

  z = z - N * x - z * M + N * x * M;

  return z;

}

