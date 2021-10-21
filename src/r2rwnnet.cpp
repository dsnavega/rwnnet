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


//[[Rcpp::export(.fit_r2rwnnet)]]
Rcpp::List fit_r2rwnnet(
    arma::mat x,
    arma::mat y,
    Rcpp::NumericVector size,
    bool skip = true,
    double eta = 1.96
) {

  std::string node = "relu";
  std::string init = "tapson";

  int depth = size.length();

  Rcpp::List std_input(depth), std_dot(depth), std_activation(depth);
  Rcpp::List std_compression(depth), std_output;
  Rcpp::List readout;
  Rcpp::List W(depth), R(depth), U(depth), B(depth);
  arma::mat z, d, h, H, Y;
  arma::cube o, O(y.n_rows, y.n_cols, depth);

  Rcpp::LogicalVector compress(depth);
  Rcpp::NumericVector lambda(depth);

  // Input Scaling
  std_input[0] = fit_zscore_scaler(x);
  d = z = transform_zscore_scaler(std_input[0], x);

  // Output Scaling
  std_output = fit_zscore_scaler(y);
  y = transform_zscore_scaler(std_output, y);

  for (int i = 0; i < depth; i++) {

    // Initialise Weights and Bias
    W[i] = initialise_weights(d, size[i], init);

    // Compute Hidden Layer
    h = transform(gaussian_noise(d, eta), W[i]);
    std_dot[i] = fit_zscore_scaler(h);
    h = transform_zscore_scaler(std_dot[i], h);

    H = activate(gaussian_noise(h, eta), node);
    std_activation[i] = fit_zscore_scaler(H);
    H = transform_zscore_scaler(std_activation[i], H);

    // Add Skip Connections
    if (skip) {
      H.insert_cols(0, z);
    }

    // Add Compression Layer
    if (H.n_cols > x.n_rows) {
      R[i] = sparse_projection(H.n_cols, x.n_rows + 1);
      H *= as<arma::mat>(R[i]);
      std_compression[i] = fit_zscore_scaler(H);
      H = transform_zscore_scaler(std_compression[i], H);
      compress[i] = true;
    }

    // Compute Implicit Ensemble
    readout = tikhonov(gaussian_noise(H, eta), gaussian_noise(y, eta));
    B[i] = readout[0];
    lambda[i] = readout[2];

    // Collect Predictions
    O.slice(i) = as<arma::mat>(readout[1]);
    o = arma::mean(O, 2);

    // Random Recursive Stacking
    if ((i + 1) < depth) {
      // Next Layer Input
      U[i] = randn(y.n_cols, z.n_cols) * sqrt(1.0 / (y.n_cols + z.n_cols));
      d = z + (o.slice(0) * as<arma::mat>(U[i]));
      std_input[i + 1] = fit_zscore_scaler(d);
      d = transform_zscore_scaler(std_input[i + 1], d);
      d = relu(d);
    }

  }

  // Scalers
  Rcpp::List scalers = Rcpp::List::create(
    Rcpp::Named("input") = std_input,             // 0
    Rcpp::Named("dot") = std_dot,                 // 1
    Rcpp::Named("activation") = std_activation,   // 2
    Rcpp::Named("compression") = std_compression, // 3
    Rcpp::Named("output") = std_output            // 4
  );

  // Average Intermediate Layers Predictions (Cube)
  o = arma::mean(O, 2);
  Y = revert_zscore_scaler(std_output, o.slice(0));

  std::string algorithm = "r2rwnnet";

  Rcpp::List network = Rcpp::List::create(
    Rcpp::Named("scalers") = scalers,     // 0
    Rcpp::Named("W") = W,                 // 1
    Rcpp::Named("R") = R,                 // 2
    Rcpp::Named("U") = U,                 // 3
    Rcpp::Named("B") = B,                 // 4
    Rcpp::Named("depth") = depth,         // 5
    Rcpp::Named("node") = node,           // 6
    Rcpp::Named("skip") = skip,           // 7
    Rcpp::Named("compress") = compress,   // 8
    Rcpp::Named("eta") = eta,             // 9
    Rcpp::Named("lambda") = lambda,       // 10
    Rcpp::Named("loocv") = Y,             // 11
    Rcpp::Named("algorithm") = algorithm  // 12
  );

  return network;

}

//[[Rcpp::export(.predict_r2rwnnet)]]
arma::mat predict_r2rwnnet(Rcpp::List network, arma::mat x) {

  // Declaration
  arma::mat z, d, h, H, Y;
  arma::cube o, O(x.n_rows, as<mat>(network[11]).n_cols, as<int>(network[5]));
  Rcpp::LogicalVector compress = network[8];

  // Input Scaling
  d = z = transform_zscore_scaler(as<List>(as<List>(network[0])[0])[0], x);

  for (int i = 0; i < as<int>(network[5]); i++) {

    // Compute Hidden Layer
    h = transform(d, as<mat>(as<List>(network[1])[i]));
    h = transform_zscore_scaler(as<List>(as<List>(network[0])[1])[i], h);
    H = activate(h, as<std::string>(network[6]));
    H = transform_zscore_scaler(as<List>(as<List>(network[0])[2])[i], H);

    // Add Direct Connections from Input to Output Layer (Skip-Layer)
    if (network[7]) {
      H.insert_cols(0, z);
    }

    // Sparse Compression
    if (compress[i]) {
      H = H * as<mat>(as<List>(network[2])[i]);
      H = transform_zscore_scaler(as<List>(as<List>(network[0])[3])[i], H);
    }

    // Collect Predictions
    O.slice(i) = H * as<mat>(as<List>(network[4])[i]);
    o = arma::mean(O, 2);

    // Random Recursive Stacking
    if ((i + 1) < as<int>(network[5])) {
      // Next Layer Input
      d = z + (o.slice(0) * as<mat>(as<List>(network[3])[i]));
      d = transform_zscore_scaler(as<List>(as<List>(network[0])[0])[i + 1], d);
      d = relu(d);
    }

  }

  // Implicit Ensemble via Averaging over Depth
  o = arma::mean(O, 2);
  Y = revert_zscore_scaler(as<List>(network[0])[4], o.slice(0));

  return Y;

}
