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

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "RcppArmadillo.h"
using namespace arma;
using namespace Rcpp;
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

// Rectifiers
arma::mat relu(arma::mat x) {

    arma::uvec index = arma::find(x < 0.0);

    if(index.n_elem > 0) {
        x(index).fill(0.0);
    }

    return x;
}

arma::mat softplus(arma::mat x) {
    return log(1.0 + arma::exp(x));
}

arma::mat gelu(arma::mat x) {
    return x % arma::normcdf(x);
}

arma::mat swish(arma::mat x) {
    return x % (1.0 / (1.0 + arma::exp(-x)));
}

arma::mat binary(arma::mat x) {

    arma::uvec n_index = arma::find(x < 0.0);
    arma::uvec p_index = arma::find(x > 0.0);

    if (n_index.n_elem > 0) {
        x(n_index).fill(0.0);
    }

    if (p_index.n_elem > 0) {
        x(p_index).fill(1.0);
    }

    return x;
}

// Transformation Layer
arma::mat transform(arma::mat x, arma::mat w, bool bias = true) {

    if (bias) {
        int n = x.n_rows;
        arma::colvec one = arma::ones(n);
        x.insert_cols(0, one);
    }

    return x * w;
}

arma::mat activate(arma::mat x, std::string node = "linear") {

    if (node == "linear") {return x;}

        // Rectifiers
        else if (node == "relu") {
            return relu(x);
        }

        else if (node == "softplus") {
            return softplus(x);
        }

        else if (node == "gelu") {
            return gelu(x);
        }

        else if (node == "swish") {
            return swish(x);
        }

        else if (node == "binary") {
            return binary(x);
        }


    else {
        std::string msg = "Invalid activation function (node): " + node;
        Rcpp::stop(msg);
    }

}

#endif
