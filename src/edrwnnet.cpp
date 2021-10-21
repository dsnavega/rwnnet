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

//[[Rcpp::export(.fit_edrwnnet)]]
Rcpp::List fit_edrwnnet(
    arma::mat x,
    arma::mat y,
    Rcpp::NumericVector size,
    double eta = 1.96
) {

  std::string node = "relu";
  std::string initialisation = "tapson";

  int depth = size.length();

  Rcpp::List readout, network;
  Rcpp::List std_input, std_output;
  Rcpp::List std_dot(depth), std_activation(depth), std_compression(depth);
  Rcpp::List W(depth), R(depth), B(depth);
  arma::mat z, h, H, d, D;
  arma::cube Y(y.n_rows, y.n_cols, depth);

  Rcpp::LogicalVector compress(depth);
  Rcpp::NumericVector lambda(depth);

  int n_input  = x.n_cols;
  int n_output = y.n_cols;

  // Input Scaling
  std_input = fit_zscore_scaler(x);
  d = z = transform_zscore_scaler(std_input, x);

  // Output Scaling
  std_output = fit_zscore_scaler(y);
  y = transform_zscore_scaler(std_output, y);

  for (int i = 0; i < depth; i++) {

    // Initialise Weights and Bias
    W[i] = initialise_weights(gaussian_noise(d, eta), size[i], initialisation);

    // Compute Hidden Layer
    h = transform(gaussian_noise(d, eta), W[i]);
    std_dot[i] = fit_zscore_scaler(h);
    h = transform_zscore_scaler(std_dot[i], h);

    H = activate(gaussian_noise(h, eta), node);
    std_activation[i] = fit_zscore_scaler(H);
    H = transform_zscore_scaler(std_activation[i], H);

    // Add Skip Connections
    H.insert_cols(0, z);

    // Sparse Compression
    if (H.n_cols > x.n_rows) {
      R[i] = sparse_projection(H.n_cols, x.n_rows);
      H *= as<arma::mat>(R[i]);
      std_compression[i] = fit_zscore_scaler(H);
      H = transform_zscore_scaler(std_compression[i], H);
      compress[i] = true;
    }

    // Compute Implicit Ensemble (Output Layers)
    readout = tikhonov(gaussian_noise(H, eta), gaussian_noise(y, eta));
    B[i] = readout[0];
    lambda[i] = readout[2];

    // Re-Scale Output
    Y.slice(i) = revert_zscore_scaler(std_output, readout[1]);

    // Next Layer Input
    d = H;

  }

  // Scalers
  Rcpp::List scalers = Rcpp::List::create(
    Rcpp::Named("input") = std_input,
    Rcpp::Named("dot") = std_dot,
    Rcpp::Named("activation") = std_activation,
    Rcpp::Named("compression") = std_compression,
    Rcpp::Named("output") = std_output
  );

  // Soft Ensemble via Layer Averaging
  Y = arma::mean(Y, 2);

  std::string algorithm = "edrwnnet";

  network = Rcpp::List::create(
    Rcpp::Named("scalers") = scalers,     // 0
    Rcpp::Named("W") = W,                 // 1
    Rcpp::Named("R") = R,                 // 2
    Rcpp::Named("B") = B,                 // 3
    Rcpp::Named("depth") = depth,         // 4
    Rcpp::Named("node") = node,           // 5
    Rcpp::Named("compress") = compress,   // 6
    Rcpp::Named("eta") = eta,             // 7
    Rcpp::Named("lambda") = lambda,       // 8
    Rcpp::Named("n_input") = n_input,     // 9
    Rcpp::Named("n_output") = n_output,   // 10
    Rcpp::Named("loocv") = Y.slice(0),    // 11
    Rcpp::Named("algorithm") = algorithm  // 12
  );

  return network;

}


//[[Rcpp::export(.predict_edrwnnet)]]
arma::mat predict_edrwnnet(Rcpp::List network, arma::mat x) {

  // Declaration & Initialization
  int depth = network[4];
  int n_output = network[10];

  Rcpp::LogicalVector compress = network[6];

  arma::mat z, d, h, H, O;
  arma::cube Y(x.n_rows, n_output, depth);

  // Input Scaling
  d = z = transform_zscore_scaler(as<List>(network[0])[0], x);

  // Deep Component
  for (int i = 0; i < depth; i++) {

    // Compute Hidden Layer
    h = transform(d, as<mat>(as<List>(network[1])[i]));
    h = transform_zscore_scaler(as<List>(as<List>(network[0])[1])[i], h);
    H = activate(h, as<std::string>(network[5]));
    H = transform_zscore_scaler(as<List>(as<List>(network[0])[2])[i], H);

    // Add Direct Connections from Input to Output Layer (Skip-Layer)
    H.insert_cols(0, z);

    // Sparse Compression
    if (compress[i]) {
      H *= as<mat>(as<List>(network[2])[i]);
      H = transform_zscore_scaler(as<List>(as<List>(network[0])[3])[i], H);
    }

    // Layer Output
    O = H * as<mat>(as<List>(network[3])[i]);
    Y.slice(i) = revert_zscore_scaler(as<List>(network[0])[4], O);

    // Next Layer Input
    d = H;

  }

  // Soft Ensemble via Layer Averaging
  Y = arma::mean(Y,2);

  return Y.slice(0);

}
