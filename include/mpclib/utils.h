/** 
 * @file utils.h
 * @author Lena
 * @brief Common types and mathematical helper functions for MPC
 * 
 * @details
 * This header gathers frequently‑used type aliases (dense, sparse, and automatic‑differentiation
 * vectors / matrices) as well as mathematical helper functions.
 */

#pragma once

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"

namespace mpclib {
/// Column vector for autodiff. Internal autodiff types are used
using ADVec = autodiff::VectorXreal;
/// Column vector for Eigen, 32 bit floating point is used
using Vec =  Eigen::VectorX<float>;
/// Dense matrix for autodiff. 32 bit floating point is used
using Mat = Eigen::MatrixX<float>;
/// Sparse Matrix for Eigen, 32 bit floating point is used
using SPMat = Eigen::SparseMatrix<float>;

/**
 * @brief Numerically stable \f$ \operatorname{sinc} \f$ function.
 *
 * @details
 * Computes \f$ \mathrm{sinc}(x) = \sin(x)/x \f$.  
 * For \f$ |x| < 10^{-3} \f$ a 4th‑order Taylor series approximation is returned
 * to avoid catastrophic cancellation:
 * \f[
 *     \mathrm{sinc}(x) \approx 1 - \frac{x^{2}}{6}.
 * \f]
 *
 * @tparam T Arithmetic type (integral or floating point).
 * @param x Input value.
 * @return \f$ \mathrm{sinc}(x) \f$.
 */
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
static inline T sinc(T x) {
    if (std::abs(x) < 1e-3f) return 1.0f - (x*x/6.0f);
    return std::sin(x) / x;
}

/**
 * @brief Mean value of a sequence using Kahan compensated summation.
 *
 * @details
 * The function returns the arithmetic mean of the elements in @p vec while
 * mitigating the loss of significance caused by finite‑precision accumulation.
 *
 * @tparam T Arithmetic type.
 * @param vec Container with the input samples.
 * @return Mean of the samples, or @c 0 if @p vec is empty.
 */
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
static inline float average(const std::vector<T>& vec) {
    if (vec.empty()) return 0;

    // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    double sum = 0; double z = 0;
    for (const auto& val : vec) {
        double y = val - z;
        double t = sum + y;
        z = (t - sum) - y;
        sum = t;
    }
    return static_cast<float>(sum / vec.size());
}

/**
 * @brief Positive‑range modulo operation.
 *
 * Returns a value in the half‑open interval \f$ [0, y) \f$ even when @p x is negative.
 *
 * @param x Dividend.
 * @param y Divisor (period).
 * @return @p x modulo @p y mapped to positive range.
 */
static inline float modfix(float x, float y) {
    float result = fmodf(x, y);
    return result + (result < 0) * y;
}

/**
 * @brief Autodiff‑compatible @ref sinc using forward‑mode @c autodiff::real
 *
 * Identical definition to @ref sinc but operates on @c autodiff::real 
 * so that derivatives are propagated automatically.
 *
 * @param x Input value
 * @return \f$ \mathrm{sinc}(x) \f$ with autodiff support.
 */
static inline auto ad_sinc(autodiff::real x) {
    if (abs(x) < 1e-3) return autodiff::real(1.0) - (x*x/6.0);
    return sin(x) / x;
}
}