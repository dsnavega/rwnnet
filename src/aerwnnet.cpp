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

//[[Rcpp::export(.fit_aerwnnet)]]
Rcpp::List fit_aerwnnet(
    arma::mat x,
    arma::mat y,
    Rcpp::NumericVector size,
    bool skip = true,
    bool flat = true,
    double eta = 1.96
) {

  int depth = size.length();

  Rcpp::List std_input, std_output, std_compression;
  Rcpp::List U(depth), B;
  Rcpp::List encoder, network;
  arma::mat z, H, D, R, Y;
  std::string init;

  Rcpp::NumericVector lambda(depth + 1);

  // Scaling
  std_input = fit_zscore_scaler(x);
  z = transform_zscore_scaler(std_input, x);

  std_output = fit_zscore_scaler(y);
  y = transform_zscore_scaler(std_output, y);

  // Unsupervised Component
  for (int i = 0; i < depth; i++) {

    if (i == 0) {

      if (size[i] > x.n_cols) {
        init = "nguyen-widrow";
      } else {
        init = "orthonormal";
      }

      encoder = fit_encoder(x, size[i], init, eta);
      lambda[i] = encoder[6];
      H = transform_encoder(encoder, x);

      if (flat)
        D.insert_cols(0, H);

    } else {

      // Enforce skip connection from input to hidden layers, encoding process
      // always have access to raw input.
      H.insert_cols(0, z);

      if (size[i] > H.n_cols) {
        init = "nguyen-widrow";
      } else {
        init = "orthonormal";
      }

      encoder = fit_encoder(H, size[i], init, eta);
      lambda[i] = encoder[6];
      H = transform_encoder(encoder, H);

      if (flat)
        D.insert_cols(D.n_cols - 1, H);

    }

    U[i] = encoder;

  }

  // Supervised Component
  bool compress = false;

  if (flat) {

    if (skip)
      D.insert_cols(0, z);

    if (D.n_cols > x.n_rows) {
      R = sparse_projection(D.n_cols, x.n_rows);
      D *= R;
      std_compression = fit_zscore_scaler(D);
      D = transform_zscore_scaler(std_compression, D);
      compress = true;
    }

    B = tikhonov(gaussian_noise(D, eta), gaussian_noise(y, eta));

  } else {

    if (skip)
      H.insert_cols(0, z);

    if (H.n_cols > x.n_rows) {
      R = sparse_projection(H.n_cols, x.n_rows);
      H *= R;
      std_compression = fit_zscore_scaler(H);
      H = transform_zscore_scaler(std_compression, H);
      compress = true;
    }

    B = tikhonov(gaussian_noise(H, eta), gaussian_noise(y, eta));

  }

  // Re-scale Output
  Y = as<arma::mat>(B[1]);
  B[1] = revert_zscore_scaler(std_output, Y);

  // Scalers
  Rcpp::List scalers = Rcpp::List::create(
    Rcpp::Named("input") = std_input,             // 0
    Rcpp::Named("compression") = std_compression, // 1
    Rcpp::Named("output") = std_output            // 2
  );

  lambda[depth] = B[2];

  std::string algorithm = "aerwnnet";

  network = Rcpp::List::create(
    Rcpp::Named("scalers") = scalers,     // 0
    Rcpp::Named("U") = U,                 // 1
    Rcpp::Named("R") = R,                 // 2
    Rcpp::Named("B") = B[0],              // 3
    Rcpp::Named("depth") = depth,         // 4
    Rcpp::Named("skip") = skip,           // 5
    Rcpp::Named("flat") = flat,           // 6
    Rcpp::Named("compress") = compress,   // 7
    Rcpp::Named("eta") = eta,             // 8
    Rcpp::Named("lambda") = lambda,       // 9
    Rcpp::Named("loocv") = B[1],          // 10
    Rcpp::Named("algorithm") = algorithm  // 11
  );

  return network;

}


//[[Rcpp::export(.predict_aerwnnet)]]
arma::mat predict_aerwnnet(Rcpp::List network, arma::mat x) {

  // Declaration & Initialization
  arma::mat z, H, D, Y;

  int depth = network[4];
  bool skip = network[5];
  bool flat = network[6];
  bool compress = network[7];

  // Input Scaling
  z = transform_zscore_scaler(as<List>(network[0])[0], x);

  // Autoencoder Component
  for (int i = 0; i < depth; i++) {

    if (i == 0) {

      H = transform_encoder(as<List>(network[1])[i], x);

      if (flat)
        D.insert_cols(0, H);

    } else {

      H.insert_cols(0, z);
      H = transform_encoder(as<List>(network[1])[i], H);

      if (flat)
        D.insert_cols(D.n_cols - 1, H);

    }

  }

  // Readout Component
  if (flat) {

    if (skip)
      D.insert_cols(0, z);

    if (compress) {
      D *= as<arma::mat>(network[2]);
      D = transform_zscore_scaler(as<List>(network[0])[1], D);
    }

    Y = D * as<arma::mat>(network[3]);

  } else {

    if (skip)
      H.insert_cols(0, z);

    if (H.n_cols > x.n_rows) {
      H *= as<arma::mat>(network[2]);
      H = transform_zscore_scaler(as<List>(network[0])[1], H);
    }

    Y = H * as<arma::mat>(network[3]);

  }

  // Output
  Y = revert_zscore_scaler(as<List>(network[0])[2], Y);

  return Y;

}
