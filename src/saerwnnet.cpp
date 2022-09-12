// This file is part of rwnnet
//
// Copyright (C) 2022, David Senhora Navega
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

//[[Rcpp::export(.fit_saerwnnet)]]
Rcpp::List fit_saerwnnet(
    arma::mat x,
    arma::mat y,
    Rcpp::NumericVector size,
    double eta = 1.96
) {

  std::string node = "relu";
  std::string init = "tapson";

  int depth = size.length();

  int n_input  = x.n_cols;
  int n_output = y.n_cols;
  int n_cases  = x.n_rows;

  Rcpp::List scalers, encoder, network;
  Rcpp::List std_input(depth), std_dot(depth), std_activation(depth);
  Rcpp::List std_compression(depth);
  Rcpp::List W(depth), R(depth), B(depth);

  Rcpp::LogicalVector compress(depth);
  Rcpp::NumericVector lambda(depth);

  arma::mat z, h, H;
  arma::cube o, O(n_cases, n_input + n_output, depth);

  // Append Output to Input
  x.insert_cols(x.n_cols, y);

  // Start Deep Learning
  for (int i = 0; i < depth; i++) {

    // Data Scaling
    std_input[i] = fit_zscore_scaler(x);
    x = z = transform_zscore_scaler(std_input[i], x);

    // Suppress Output for Supervised Encoding Step (i = 0)
    if (i == 0) {
      z.cols(n_input, n_input + n_output - 1).fill(0.0);
    }

    // Hidden Layer Weights
    W[i] = initialise_weights(gaussian_noise(z, eta), size[i], init);

    // Compute Hidden Layer
    h = transform(gaussian_noise(z, eta), W[i]);
    std_dot[i] = fit_zscore_scaler(h);
    h = transform_zscore_scaler(std_dot[i], h);

    H = activate(gaussian_noise(h, eta), node);
    std_activation[i] = fit_zscore_scaler(H);
    H = transform_zscore_scaler(std_activation[i], H);

    if (size[i] > n_cases) {
      R[i] = sparse_projection(size[i], n_cases);
      H *= as<arma::mat>(R[i]);
      std_compression[i] = fit_zscore_scaler(H);
      H = transform_zscore_scaler(std_compression[i], H);
      compress[i] = true;
    }

    // Tikhonov and Gaussian Noise Regularization
    encoder = tikhonov(gaussian_noise(H, eta), x);
    encoder[1] = revert_zscore_scaler(std_input[0], encoder[1]);
    // Read-out weights
    B[i] = encoder[0];

    // Regularization
    lambda[i] = encoder[2];

    // Next Layer Input
    x = as<arma::mat>(encoder[1]);

    // Soft Ensemble via Layer Averaging
    O.slice(i) = x;
    o = arma::mean(O, 2);
    x = o.slice(0);

  }

  // Scalers
  scalers = Rcpp::List::create(
    Rcpp::Named("input") = std_input,             // 0
    Rcpp::Named("dot") = std_dot,                 // 1
    Rcpp::Named("activation") = std_activation,   // 2
    Rcpp::Named("compression") = std_compression  // 3
  );

  // Reconstructed (Denoised) and Predicted Input and Output
  x = o.slice(0).cols(0, n_input - 1);
  y = o.slice(0).cols(n_input, n_input + n_output - 1);

  std::string algorithm = "saerwnnet";

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
    Rcpp::Named("reconstructed") = x,     // 11
    Rcpp::Named("loocv") = y,             // 12
    Rcpp::Named("algorithm") = algorithm  // 13
  );

  return network;

}

//[[Rcpp::export(.predict_saerwnnet)]]
arma::mat predict_saerwnnet(Rcpp::List network, arma::mat x) {

  // Declaration and Initialization
  arma::mat z, h, H;
  int n_input = network[9];
  int n_output = network[10];
  int depth = network[4];
  Rcpp::LogicalVector compress = network[6];
  arma::cube o, O(x.n_rows, n_input + n_output, depth);

  // Initialize Columns for Output
  arma::mat Y(x.n_rows, n_output);
  Y.fill(1.0);

  // Append Output to Input
  x.insert_cols(n_input, Y);

  for (int i = 0; i < depth; i++) {

    // Data Scaling
    z = transform_zscore_scaler(as<List>(as<List>(network[0])[0])[i], x);

    // Suppress Output for Supervised Encoding Step (i = 0)
    if (i == 0) {
      z.cols(n_input, (n_input + n_output) - 1).fill(0.0);
    }

    // Compute Hidden Layer
    h = transform(z, as<mat>(as<List>(network[1])[i]));
    h = transform_zscore_scaler(as<List>(as<List>(network[0])[1])[i], h);
    H = activate(h, as<std::string>(network[5]));
    H = transform_zscore_scaler(as<List>(as<List>(network[0])[2])[i], H);

    // Sparse Compression
    if (compress[i]) {
      H *= as<mat>(as<List>(network[2])[i]);
      H = transform_zscore_scaler(as<List>(as<List>(network[0])[3])[i], H);
    }

    // De-Noised Input and Predicted Output
    x = H * as<mat>(as<List>(network[3])[i]);
    x = revert_zscore_scaler(as<List>(as<List>(network[0])[0])[0], x);
    O.slice(i) = x;

    // Next Layer
    o = arma::mean(O, 2);
    x = o.slice(0);

  }

  // Implicit Ensemble via Averaging over Depth
  Y = o.slice(0).cols(n_input, n_input + n_output - 1);
  return Y;

}
