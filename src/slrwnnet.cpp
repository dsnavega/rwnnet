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

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

#include "weights.h"
#include "activation.h"
#include "utilities.h"
#include "tikhonov.h"

//[[Rcpp::export(.fit_slrwnnet)]]
Rcpp::List fit_slrwnnet(
    arma::mat x,
    arma::mat y,
    int size,
    bool skip = true,
    double eta = 1.96
) {

  std::string node = "relu";
  std::string init = "tapson";

  Rcpp::List std_input, std_dot, std_activation;
  Rcpp::List std_compression, std_output;
  Rcpp::List readout;
  arma::mat z, W, h, H, R, Y;

  // Input Scaling
  std_input = fit_zscore_scaler(x);
  z = transform_zscore_scaler(std_input, x);

  // Output Scaling
  std_output = fit_zscore_scaler(y);
  y = transform_zscore_scaler(std_output, y);

  // Initialise Random Weights and Bias
  W = initialise_weights(gaussian_noise(z, eta), size, init);

  // Compute Hidden Layer
  h = transform(gaussian_noise(z, eta), W);
  std_dot = fit_zscore_scaler(h);
  h = transform_zscore_scaler(std_dot, h);

  H = activate(gaussian_noise(h, eta), node);
  std_activation = fit_zscore_scaler(H);
  H = transform_zscore_scaler(std_activation, H);

  // Add Direct Connections from Input to Output Layer (Skip-Layer)
  if (skip)
    H.insert_cols(0, z);

  bool compress = false;
  if (H.n_cols > x.n_rows) {
    R = sparse_projection(H.n_cols, x.n_rows);
    H = H * R;
    std_compression = fit_zscore_scaler(H);
    H = transform_zscore_scaler(std_compression, H);
    compress = true;
  }

  // Tikhonov and Gaussian Noise Regularization
  readout = tikhonov(gaussian_noise(H, eta), gaussian_noise(y, eta));

  // Scalers
  Rcpp::List scalers = Rcpp::List::create(
    Rcpp::Named("input") = std_input,
    Rcpp::Named("dot") = std_dot,
    Rcpp::Named("activation") = std_activation,
    Rcpp::Named("compression") = std_compression,
    Rcpp::Named("output") = std_output
  );

  // Re-Scale Output
  Y = as<arma::mat>(readout[1]);
  readout[1] = revert_zscore_scaler(std_output, Y);

  std::string algorithm = "slrwnnet";

  Rcpp::List network = Rcpp::List::create(
    Rcpp::Named("scalers") = scalers,
    Rcpp::Named("W") = W,
    Rcpp::Named("R") = R,
    Rcpp::Named("B") = readout[0],
    Rcpp::Named("node") = node,
    Rcpp::Named("skip") = skip,
    Rcpp::Named("compress") = compress,
    Rcpp::Named("eta") = eta,
    Rcpp::Named("lambda") = readout[2],
    Rcpp::Named("loocv") = readout[1],
    Rcpp::Named("algorithm") = algorithm
  );

  return network;

}

//[[Rcpp::export(.predict_slrwnnet)]]
arma::mat predict_slrwnnet(Rcpp::List network, arma::mat x) {

  // Declaration
  arma::mat z, h, H, Y;

  // Input Scaling
  z = transform_zscore_scaler(as<List>(network[0])[0], x);

  // Compute Hidden Layer
  h = transform(z, as<arma::mat>(network[1]));
  h = transform_zscore_scaler(as<List>(network[0])[1], h);
  H = activate(h, as<std::string>(network[4]));
  H = transform_zscore_scaler(as<List>(network[0])[2], H);

  // Add Direct Connections from Input to Output Layer (Skip-Layer)
  if (network[5])
    H.insert_cols(0, z);

  // Sparse Compression
  if (network[6]) {
    H *= as<arma::mat>(network[2]);
    H = transform_zscore_scaler(as<List>(network[0])[3], H);
  }

  // Readout (Prediction)
  Y = H * as<arma::mat>(network[3]);
  return revert_zscore_scaler(as<List>(network[0])[4], Y);

}
