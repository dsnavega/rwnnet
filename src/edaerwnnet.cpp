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
#include "encoder.h"

//[[Rcpp::export(.fit_edaerwnnet)]]
Rcpp::List fit_edaerwnnet(
    arma::mat x,
    arma::mat y,
    Rcpp::NumericVector size,
    bool skip = true,
    double eta = 1.96
) {

  int depth = size.length();
  int n_input  = x.n_cols;
  int n_output = y.n_cols;

  std::string init;

  Rcpp::List std_input, std_output, encoder, readout, network;
  Rcpp::List std_compression(depth);
  Rcpp::List U(depth), R(depth), B(depth);
  arma::mat z, H, D;
  arma::cube Y(y.n_rows, y.n_cols, depth);

  Rcpp::LogicalVector compress(depth);
  Rcpp::NumericVector u_lambda(depth);
  Rcpp::NumericVector r_lambda(depth);

  // Input Scaling
  std_input = fit_zscore_scaler(x);
  z = transform_zscore_scaler(std_input, x);

  // Output Scaling
  std_output = fit_zscore_scaler(y);
  y = transform_zscore_scaler(std_output, y);

  for (int i = 0; i < depth; i++) {

    // Unsupervised Component
    if (i == 0) {

      if (size[i] > x.n_cols) {
        init = "nguyen-widrow";
      } else {
        init = "orthonormal";
      }
      encoder = fit_encoder(x, size[i], init, eta);
      H = transform_encoder(encoder, x);
      u_lambda[i] = encoder[6];
      U[i] = encoder;

    } else {

      H.insert_cols(0, z);

      if (size[i] > x.n_cols) {
        init = "nguyen-widrow";
      } else {
        init = "orthonormal";
      }

      encoder = fit_encoder(H, size[i], init, eta);
      H = transform_encoder(encoder, H);
      u_lambda[i] = encoder[6];
      U[i] = encoder;

    }

    // Supervised Component
    if (skip)
      H.insert_cols(0, z);

    if (H.n_cols > x.n_rows) {
      R[i] = sparse_projection(H.n_cols, x.n_rows);
      H *= as<arma::mat>(R[i]);
      std_compression = fit_zscore_scaler(H);
      H = transform_zscore_scaler(std_compression, H);
      compress[i] = true;
    }

    // Tikhonov Regularization and Gaussian noise.
    readout = tikhonov(gaussian_noise(H, eta), gaussian_noise(y, eta));
    r_lambda[i] = readout[2];
    B[i] = readout[0];

    // Output
    Y.slice(i) = as<arma::mat>(readout[1]);
    Y.slice(i) = revert_zscore_scaler(std_output, Y.slice(i));

  }

  // Average Intermediate Layers Predictions (Cube)
  Y = arma::mean(Y, 2);

  // Scalers
  Rcpp::List scalers = Rcpp::List::create(
    Rcpp::Named("input") = std_input,
    Rcpp::Named("compression") = std_compression,
    Rcpp::Named("output") = std_output
  );

  std::string algorithm = "edaerwnnet";

  network = Rcpp::List::create(
    Rcpp::Named("scalers") = scalers,           // 0
    Rcpp::Named("U") = U,                       // 1
    Rcpp::Named("R") = R,                       // 2
    Rcpp::Named("B") = B,                       // 3
    Rcpp::Named("depth") = depth,               // 4
    Rcpp::Named("skip") = skip,                 // 5
    Rcpp::Named("compress") = compress,         // 6
    Rcpp::Named("lambda") = Rcpp::List::create( // 7
      Rcpp::Named("unsupervised") = u_lambda,
      Rcpp::Named("supervised") = r_lambda
    ),
    Rcpp::Named("eta") = eta,                   // 8
    Rcpp::Named("n_input") = n_input,           // 9
    Rcpp::Named("n_output") = n_output,         // 10
    Rcpp::Named("loocv") = Y.slice(0),          // 11
    Rcpp::Named("algorithm") = algorithm        // 12
  );

  return network;

}

//[[Rcpp::export(.predict_edaerwnnet)]]
arma::mat predict_edaerwnnet(Rcpp::List network, arma::mat x) {

  // Declaration & Initialization
  int depth = network[4];
  bool skip = network[5];

  Rcpp::LogicalVector compress = network[6];
  int n_output = network[10];

  arma::mat z, H;
  arma::cube Y(x.n_rows, n_output,depth);

  // Input Scaling
  z = transform_zscore_scaler(as<List>(network[0])[0], x);

  // Autoencoder Component
  for (int i = 0; i < depth; i++) {

    if (i == 0) {

      H = transform_encoder(as<List>(network[1])[i], x);

    } else {

      H.insert_cols(0, z);
      H = transform_encoder(as<List>(network[1])[i], H);

    }

    if (skip)
      H.insert_cols(0, z);

    if (compress[i]) {
      H *= as<mat>(as<List>(network[2])[i]);
      H = transform_zscore_scaler(as<List>(as<List>(network[0])[1])[i], H);
    }

    Y.slice(i) = H * as<mat>(as<List>(network[3])[i]);
    Y.slice(i) = revert_zscore_scaler(as<List>(network[0])[2], Y.slice(i));

  }

  // Output
  Y = arma::mean(Y, 2);
  return Y.slice(0);

}
