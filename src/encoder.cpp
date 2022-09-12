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

Rcpp::List fit_encoder(
    arma::mat x,
    int size,
    double eta = 1.96
) {

  // Rectifier
  std::string node = "relu";
  std::string init = "nguyen-widrow";

  // Declare Variables
  Rcpp::List readout;
  Rcpp::List scalers;
  Rcpp::List std_input;
  Rcpp::List std_dot_io, std_activation_io;
  Rcpp::List std_dot_oi, std_activation_oi;
  Rcpp::List std_compression;
  arma::mat W, z, h, H, R, B;

  // Input Scaling
  std_input = fit_zscore_scaler(x);
  z = transform_zscore_scaler(std_input, x);

  // Initialise Random Weights
  W = initialise_weights(z, size, init);

  // Compute Hidden Layer
  h = transform(gaussian_noise(z, eta), W);
  std_dot_io = fit_zscore_scaler(h);
  h = transform_zscore_scaler(std_dot_io, h);

  H = activate(gaussian_noise(h, eta), node);
  std_activation_io = fit_zscore_scaler(H);
  H = transform_zscore_scaler(std_activation_io, H);

  bool compress = false;
  if (H.n_cols > x.n_rows) {
    R = sparse_projection(H.n_cols, x.n_rows);
    H *= R;
    std_compression = fit_zscore_scaler(H);
    H = transform_zscore_scaler(std_compression, H);
    compress = true;
  }

  // Learn Encoding via Tikhonov Regularization
  readout = tikhonov(gaussian_noise(H, eta), z);
  B = as<arma::mat>(readout[0]);

  // Update Scalers (Dot & Activation) (for Stacking)
  h = transform(z, B.t(), false);
  std_dot_oi = fit_zscore_scaler(h);
  h = transform_zscore_scaler(std_dot_oi, h);
  H = activate(h, node);

  std_activation_oi = fit_zscore_scaler(H);
  H = transform_zscore_scaler(std_activation_oi, H);

  // Scalers
  scalers = Rcpp::List::create(
    Rcpp::Named("input") = std_input,               // 0
    Rcpp::Named("dot") = Rcpp::List::create(        // 1
      std_dot_io,
      std_dot_oi
    ),
    Rcpp::Named("activation") = Rcpp::List::create( // 2
      std_activation_io,
      std_activation_oi
    ),
    Rcpp::Named("compression") = std_compression    // 3
  );

  Rcpp::List encoder = Rcpp::List::create(
    Rcpp::Named("scalers") = scalers,       // 0
    Rcpp::Named("W") = W,                   // 1
    Rcpp::Named("R") = R,                   // 2
    Rcpp::Named("B") = B,                   // 3
    Rcpp::Named("node") = node,             // 4
    Rcpp::Named("compress") = compress,     // 5
    Rcpp::Named("lambda") = readout[2],     // 6
    Rcpp::Named("loocv") = readout[1]       // 7
  );

  return encoder;

}

arma::mat transform_encoder(
    Rcpp::List encoder,
    arma::mat x
) {

  // Declare Variables
  arma::mat z, h, H;
  Rcpp::List std_dot, std_activation;

  // Input Scaling
  z = transform_zscore_scaler(as<List>(encoder[0])[0], x);

  // Compute Hidden Layer
  std_dot = as<List>(encoder[0])[1];
  std_activation = as<List>(encoder[0])[2];

  h = transform(z, as<arma::mat>(encoder[3]).t(), false);
  h = transform_zscore_scaler(std_dot[1], h);
  H = activate(h, encoder[4]);
  H = transform_zscore_scaler(std_activation[1], H);

  return H;

}


arma::mat reconstruct_encoder(
    Rcpp::List encoder,
    arma::mat x
) {

  // Declare Variables
  arma::mat z, h, H, Y;
  Rcpp::List std_dot, std_activation;

  // Input Scaling
  z = transform_zscore_scaler(as<List>(encoder[0])[0], x);

  // Compute Hidden Layer
  std_dot = as<List>(encoder[0])[1];
  std_activation = as<List>(encoder[0])[2];

  h = transform(z, as<arma::mat>(encoder[1]), true);
  h = transform_zscore_scaler(std_dot[0], h);
  H = activate(h, encoder[4]);
  H = transform_zscore_scaler(std_activation[0], H);

  // Sparse Compression
  if (encoder[5]) {
    H = H * as<arma::mat>(encoder[2]);
    H = transform_zscore_scaler(as<List>(encoder[0])[3], H);
  }

  // Re-construct and Re-scale
  Y = H * as<arma::mat>(encoder[3]);
  Y = revert_zscore_scaler(as<List>(encoder[0])[0], Y);

  return Y;

}
