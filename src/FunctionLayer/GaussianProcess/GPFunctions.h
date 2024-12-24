#pragma once
#include "CoreLayer/Adapter/JsonUtil.h"
#include "CoreLayer/Geometry/Geometry.h"
#include "CoreLayer/Geometry/Matrix.h"

#include "FunctionLayer/SDFFunction/SdfFunctions.h"

#include "autodiff/forward/dual.hpp"
#include "autodiff/forward/real.hpp"
#include "autodiff/forward/dual/eigen.hpp"
#include "autodiff/forward/real/eigen.hpp"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/examples/benchmark/DenseGrid.h>

#include "GaussianProcessUtils.h"

class MeanFunction {
public:
    MeanFunction() = default;
    MeanFunction(const Json &json);
    virtual double operator()(const DerivativeType &derivativeType, const Point3d &point, const Vec3d &derivativeDir = {}) const {
        if (derivativeType == DerivativeType::None) {
            return mean(point);
        } else {
            return dot(dmean_dp(point), derivativeDir);
        }
    }

protected:
    virtual double mean(const Point3d &point) const = 0;
    virtual Vec3d dmean_dp(const Point3d &point) const;

    // Uniform scaling across all axes is required
    double transformScale;
    mutable TransformMatrix3D transformMatrix;
    mutable TransformMatrix3D invTransformMatrix;

    double scale;
    double offset;
};

class ProceduralMean : public MeanFunction {
public:
    ProceduralMean(const Json &json);

protected:
    virtual double mean(const Point3d &point) const override;
    SdfFunctions::Function func;
};

class TabulatedMean : public MeanFunction {
public:
    TabulatedMean(const Json &json);

protected:
    virtual double mean(const Point3d &point) const override;

    std::string meanGridName;
    nanovdb::DenseGridHandle<nanovdb::HostBuffer> meanGrid;
    nanovdb::DenseGrid<float> *meanFloatGrid = nullptr;

    bool gridShouldBeNormalized = false;
};

class CovarianceFunction {
public:
    CovarianceFunction() = default;
    virtual double operator()(const DerivativeType &derivativeTypeX, const Point3d &pointX,
                              const DerivativeType &derivativeTypeY, const Point3d &pointY,
                              const Vec3d &derivativeDirX = {}, const Vec3d &derivativeDirY = {}) const {
        if (derivativeTypeX == DerivativeType::None) {
            if (derivativeTypeY == DerivativeType::None) {
                return cov(pointX, pointY);
            } else {
                return dcov_dy(pointX, pointY, derivativeDirY);
            }
        } else {
            if (derivativeTypeY == DerivativeType::None) {
                return dcov_dx(pointX, pointY, derivativeDirX);
            } else {
                return dcov2_dxdy(pointX, pointY, derivativeDirX, derivativeDirY);
            }
        }
        return 0.;
    }

protected:
    // common interface
    virtual double cov(const Point3d &pointX, const Point3d &pointY) const = 0;
    // interfaces for autodiff
    virtual autodiff::real2nd cov(const autodiff::Vector3real2nd &pointX, const autodiff::Vector3real2nd &pointY) const = 0;
    virtual autodiff::dual2nd cov(const autodiff::Vector3dual2nd &pointX, const autodiff::Vector3dual2nd &pointY) const = 0;

    double dcov_dx(const Point3d &pointX, const Point3d &pointY, const Vec3d &ddirX) const;
    double dcov_dy(const Point3d &pointX, const Point3d &pointY, const Vec3d &ddirY) const;
    double dcov2_dxdy(const Point3d &pointX, const Point3d &pointY, const Vec3d &ddirX, const Vec3d &ddirY) const;
};

class StationaryCovariance : public CovarianceFunction {
    friend class NonstationaryCovariance;

protected:
    virtual autodiff::real2nd cov(const autodiff::real2nd &dis2) const = 0;
    virtual autodiff::dual2nd cov(const autodiff::dual2nd &dis2) const = 0;
    virtual double cov(double dis) const = 0;

    virtual double cov(const Point3d &pointX, const Point3d &pointY) const {
        auto d = pointX - pointY;
        return cov(dot(d, d));
    }
    virtual autodiff::real2nd cov(const autodiff::Vector3real2nd &pointX, const autodiff::Vector3real2nd &pointY) const {
        auto d = pointX - pointY;
        return cov(d.dot(d));
    }
    virtual autodiff::dual2nd cov(const autodiff::Vector3dual2nd &pointX, const autodiff::Vector3dual2nd &pointY) const {
        auto d = pointX - pointY;
        return cov(d.dot(d));
    }
};

class SquaredExponentialCovariance : public StationaryCovariance {
public:
    SquaredExponentialCovariance(const Json &json);

protected:
    virtual autodiff::real2nd cov(const autodiff::real2nd &dis2) const override;
    virtual autodiff::dual2nd cov(const autodiff::dual2nd &dis2) const override;
    virtual double cov(double dis2) const;

protected:
    double sigma;
    double lengthScale;
};

class NonstationaryCovariance : public CovarianceFunction {
public:
    NonstationaryCovariance(const Json &json);

protected:
    virtual double cov(const Point3d &pointX, const Point3d &pointY) const override;
    virtual autodiff::real2nd cov(const autodiff::Vector3real2nd &pointX, const autodiff::Vector3real2nd &pointY) const override;
    virtual autodiff::dual2nd cov(const autodiff::Vector3dual2nd &pointX, const autodiff::Vector3dual2nd &pointY) const override;

    mutable TransformMatrix3D transformMatrix;
    mutable TransformMatrix3D invTransformMatrix;

    double sampleLocalVariance(const Point3d &point) const;
    autodiff::real2nd sampleLocalVariance(const autodiff::Vector3real2nd &point) const;
    autodiff::dual2nd sampleLocalVariance(const autodiff::Vector3dual2nd &point) const;

    std::string localVarianceGridName;
    nanovdb::DenseGridHandle<nanovdb::HostBuffer> localVarianceGrid;
    nanovdb::DenseGrid<float> *localVarianceFloatGrid = nullptr;

    Point3d clamp(Point3d point, const nanovdb::BBoxR *bbox) const {
        return {
            std::clamp(point.x, bbox->min()[0], bbox->max()[0]),
            std::clamp(point.y, bbox->min()[1], bbox->max()[1]),
            std::clamp(point.z, bbox->min()[2], bbox->max()[2]),
        };
    }

    // TODO(Cchen77): a spatial varied local anisotropy: R3 -> 3x3 PSD
    // for now we just use a scalar to describe correlation without anisotropy

    double sampleCorrelation(const Point3d &point) const;
    autodiff::real2nd sampleCorrelation(const autodiff::Vector3real2nd &point) const;
    autodiff::dual2nd sampleCorrelation(const autodiff::Vector3dual2nd &point) const;

    std::string correlationGridName;
    nanovdb::DenseGridHandle<nanovdb::HostBuffer> correlationGrid;
    nanovdb::DenseGrid<float> *correlationFloatGrid = nullptr;

    bool gridShouldBeNormalized = false;

    std::shared_ptr<CovarianceFunction> stationaryCovariance;
};

inline float sampleDenseGrid(const nanovdb::DenseGrid<float> &denseGrid, const Point3d &p) {
    double eps = 1e-8;
    Point3d point = p;

    if (!denseGrid.worldBBox().isInside({p[0], p[1], p[2]})) {
        auto bbox = denseGrid.worldBBox();
        point = {std::clamp(point.x, bbox.min()[0] + eps, bbox.max()[0] - eps),
                 std::clamp(point.y, bbox.min()[1] + eps, bbox.max()[1] - eps),
                 std::clamp(point.z, bbox.min()[2] + eps, bbox.max()[2] - eps)};
    }

    Point3d index = denseGrid.worldToIndex(point);
    auto &indexMax = denseGrid.indexBBox().max();
    auto &indexMin = denseGrid.indexBBox().min();

    int x0 = fm::floor(index[0]);
    int y0 = fm::floor(index[1]);
    int z0 = fm::floor(index[2]);

    double u = fm::abs(x0 + 0.5 - index[0]);
    double v = fm::abs(y0 + 0.5 - index[1]);
    double w = fm::abs(z0 + 0.5 - index[2]);

    double one_minus_u = 1.0 - u;
    double one_minus_v = 1.0 - v;
    double one_minus_w = 1.0 - w;

    auto getVoxel = [&](int x, int y, int z) -> float {
        return denseGrid.getValue({std::clamp(x, indexMin[0], indexMax[0]),
                                   std::clamp(y, indexMin[1], indexMax[1]),
                                   std::clamp(z, indexMin[2], indexMax[2])});
    };

    float c000 = getVoxel(x0, y0, z0);
    float c100 = getVoxel((index[0] - x0 > 0.5) ? x0 + 1 : x0 - 1, y0, z0);
    float c010 = getVoxel(x0, (index[1] - y0 > 0.5) ? y0 + 1 : y0 - 1, z0);
    float c001 = getVoxel(x0, y0, (index[2] - z0 > 0.5) ? z0 + 1 : z0 - 1);
    float c110 = getVoxel((index[0] - x0 > 0.5) ? x0 + 1 : x0 - 1, (index[1] - y0 > 0.5) ? y0 + 1 : y0 - 1, z0);
    float c101 = getVoxel((index[0] - x0 > 0.5) ? x0 + 1 : x0 - 1, y0, (index[2] - z0 > 0.5) ? z0 + 1 : z0 - 1);
    float c011 = getVoxel(x0, (index[1] - y0 > 0.5) ? y0 + 1 : y0 - 1, (index[2] - z0 > 0.5) ? z0 + 1 : z0 - 1);
    float c111 = getVoxel((index[0] - x0 > 0.5) ? x0 + 1 : x0 - 1, (index[1] - y0 > 0.5) ? y0 + 1 : y0 - 1, (index[2] - z0 > 0.5) ? z0 + 1 : z0 - 1);

    return one_minus_u * one_minus_v * one_minus_w * c000 +
           u * one_minus_v * one_minus_w * c100 +
           one_minus_u * v * one_minus_w * c010 +
           one_minus_u * one_minus_v * w * c001 +
           u * one_minus_v * w * c101 +
           one_minus_u * v * w * c011 +
           u * v * one_minus_w * c110 +
           u * v * w * c111;
}
