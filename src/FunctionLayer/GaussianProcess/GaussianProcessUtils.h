#pragma once
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "CoreLayer/Geometry/Geometry.h"
#include "FunctionLayer/Sampler/Sampler.h"
#include "CoreLayer/Math/Common.h"

#define SQRT_2 1.4142135623730951
#define SQRT_2PI 2.5066282746310007

enum class DerivativeType {
    None,
    First,
};

inline DerivativeType *DerivativeTypeNone() {
    static DerivativeType type = DerivativeType::None;
    return &type;
}
inline DerivativeType *DerivativeTypeFirst() {
    static DerivativeType type = DerivativeType::First;
    return &type;
}

template<typename To, typename From>
inline To vec_conv(const From &vd) {
    return To{vd.x, vd.y, vd.x};
}
// Box muller transform
inline Vec2d rand_normal_2(Sampler &sampler) {
    double u1 = sampler.sample1D();
    double u2 = sampler.sample1D();

    double r = fm::sqrt(-2 * log(1. - u1));
    double x = fm::cos(2 * M_PI * u2);
    double y = fm::sin(2 * M_PI * u2);
    double z1 = r * x;
    double z2 = r * y;

    return Vec2d(z1, z2);
}

// https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
inline double my_erfinvf(double a) {
    double p, r, t;
    t = std::fma(a, .0 - a, 1.);
    t = std::log(t);
    if (std::fabs(t) > 6.125) {            // maximum ulp error = 2.35793
        p = 3.03697567e-10f;               //  0x1.4deb44p-32
        p = std::fma(p, t, 2.93243101e-8); //  0x1.f7c9aep-26
        p = std::fma(p, t, 1.22150334e-6); //  0x1.47e512p-20
        p = std::fma(p, t, 2.84108955e-5); //  0x1.dca7dep-16
        p = std::fma(p, t, 3.93552968e-4); //  0x1.9cab92p-12
        p = std::fma(p, t, 3.02698812e-3); //  0x1.8cc0dep-9
        p = std::fma(p, t, 4.83185798e-3); //  0x1.3ca920p-8
        p = std::fma(p, t, -2.64646143e-1);// -0x1.0eff66p-2
        p = std::fma(p, t, 8.40016484e-1); //  0x1.ae16a4p-1
    } else {                               // maximum ulp error = 2.35002
        p = 5.43877832e-9;                 //  0x1.75c000p-28
        p = std::fma(p, t, 1.43285448e-7); //  0x1.33b402p-23
        p = std::fma(p, t, 1.22774793e-6); //  0x1.499232p-20
        p = std::fma(p, t, 1.12963626e-7); //  0x1.e52cd2p-24
        p = std::fma(p, t, -5.61530760e-5);// -0x1.d70bd0p-15
        p = std::fma(p, t, -1.47697632e-4);// -0x1.35be90p-13
        p = std::fma(p, t, 2.31468678e-3); //  0x1.2f6400p-9
        p = std::fma(p, t, 1.15392581e-2); //  0x1.7a1e50p-7
        p = std::fma(p, t, -2.32015476e-1);// -0x1.db2aeep-3
        p = std::fma(p, t, 8.86226892e-1); //  0x1.c5bf88p-1
    }
    r = a * p;
    return r;
}

inline double gaussianCDF(double mu, double s, double x) {
    return 0.5 * std::erfc(-(x - mu) / (s * SQRT_2));
}

inline double gaussianPDF(double mu, double s, double x) {
    return std::exp(-std::pow(x - mu, 2.) / (2 * s * s)) / (s * SQRT_2PI);
}

inline double gaussianQuantile(double mu, double s, double p) {
    return mu - s * sqrt(2) * my_erfinvf(2 * p);
}

struct MultiVariableNormalDistribution {
    Eigen::VectorXd mean;

    Eigen::BDCSVD<Eigen::MatrixXd> svd;

    Eigen::MatrixXd normTransform;

    MultiVariableNormalDistribution(const Eigen::VectorXd &_mean, const Eigen::MatrixXd &_cov);

    Eigen::VectorXd sample(Sampler &sampler) const;
};
