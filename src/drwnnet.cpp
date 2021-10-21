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

//[[Rcpp::export(.fit_drwnnet)]]
Rcpp::List fit_drwnnet(
    arma::mat x,
    arma::mat y,
    Rcpp::NumericVector size,
    double eta = 1.96
) {


  std::string node = "relu";
  std::string init = "tapson";

  int depth = size.length();

  Rcpp::List readout, network;
  Rcpp::List std_input, std_compression, std_output;
  Rcpp::List std_dot(depth), std_activation(depth);
  Rcpp::List W(depth);
  arma::mat z, d, h, H, D, R, B, Y;

  // Input Scaling
  std_input = fit_zscore_scaler(x);
  d = z = transform_zscore_scaler(std_input, x);

  // Output Scaling
  std_output = fit_zscore_scaler(y);
  y = transform_zscore_scaler(std_output, y);

  for(int i = 0; i < depth; i++) {

    // Initialise Weights and Bias
    W[i] = initialise_weights(gaussian_noise(d, eta), size[i], init);

    // Compute Hidden Layer
    h = transform(gaussian_noise(d, eta), W[i]);
    std_dot[i] = fit_zscore_scaler(h);
    h = transform_zscore_scaler(std_dot[i], h);

    H = activate(gaussian_noise(h, eta), node);
    std_activation[i] = fit_zscore_scaler(H);
    H = transform_zscore_scaler(std_activation[i], H);

    if (i == 0) {
      D = H;
    } else {
      D.insert_cols(D.n_cols - 1, H);
    }

    // Next Layer Input
    d = H;
    d.insert_cols(0, z);

  }

  // Add Direct Connections from Input to Output Layer (Skip-Layer)
  D.insert_cols(0, z);

  // Sparse Compression
  bool compress = false;
  if (D.n_cols > x.n_rows) {
    R = sparse_projection(D.n_cols, x.n_rows);
    D = D * R;
    std_compression = fit_zscore_scaler(D);
    D = transform_zscore_scaler(std_compression, D);
    compress = true;
  }

  // Tikhonov and Gaussian Noise Regularization
  readout = tikhonov(gaussian_noise(D, eta), gaussian_noise(y, eta));

  // Re-scale Output
  Y = revert_zscore_scaler(std_output, as<arma::mat>(readout[1]));

  // Scalers
  Rcpp::List scalers = Rcpp::List::create(
    Rcpp::Named("input") = std_input,             // 0
    Rcpp::Named("dot") = std_dot,                 // 1
    Rcpp::Named("activation") = std_activation,   // 2
    Rcpp::Named("compression") = std_compression, // 3
    Rcpp::Named("output") = std_output            // 4
  );

  std::string algorithm = "drwnnet";

  network = Rcpp::List::create(
    Rcpp::Named("scalers") = scalers,     // 0
    Rcpp::Named("W") = W,                 // 1
    Rcpp::Named("R") = R,                 // 2
    Rcpp::Named("B") = readout[0],        // 3
    Rcpp::Named("depth") = depth,         // 4
    Rcpp::Named("node") = node,           // 5
    Rcpp::Named("compress") = compress,   // 6
    Rcpp::Named("eta") = eta,             // 7
    Rcpp::Named("lambda") = readout[2],   // 8
    Rcpp::Named("loocv") = Y,             // 9
    Rcpp::Named("algorithm") = algorithm  // 10
  );

  return network;

}

//[[Rcpp::export(.predict_drwnnet)]]
arma::mat predict_drwnnet(Rcpp::List network, arma::mat x) {

  // Declaration & Initialization
  arma::mat z, d, h, H, D, Y;
  int depth = network[4];
  bool compress = network[6];

  // Input Scaling
  d = z = transform_zscore_scaler(as<List>(network[0])[0], x);

  // Deep Transformation
  for (int i = 0; i < depth; i++) {

    // Compute Hidden Layer
    h = transform(d, as<mat>(as<List>(network[1])[i]));
    h = transform_zscore_scaler(as<List>(as<List>(network[0])[1])[i], h);
    H = activate(h, as<std::string>(network[5]));
    H = transform_zscore_scaler(as<List>(as<List>(network[0])[2])[i], H);

    // Append Current Layer
    if (i == 0) {
      D = H;
    } else {
      D.insert_cols(D.n_cols - 1, H);
    }

    // Next Layer Input
    d = H;
    d.insert_cols(0, z);

  }

  // Add Direct Connections from Input to Output Layer (Skip-Layer)
  D.insert_cols(0, z);

  // Sparse Compression
  if (compress) {
    D *= as<mat>(network[2]);
    D = transform_zscore_scaler(as<List>(network[0])[3], D);
  }

  // Output
  Y = D * as<arma::mat>(network[3]);
  Y = revert_zscore_scaler(as<List>(network[0])[4], Y);

  return Y;

}

