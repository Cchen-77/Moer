#pragma once

#include "Medium.h"
#include "CoreLayer/Adapter/JsonUtil.h"
#include "FunctionLayer/GaussianProcess/GaussianProcess.h"
#define GPIS_LIGHT_TRANSPORT_VERSION 2
class GPISMedium : public Medium {
public:
    GPISMedium() = default;
    GPISMedium(const Json &json);

    virtual bool sampleDistance(MediumSampleRecord *mRec,
                                const Ray &ray,
                                const Intersection &its,
                                Point2d sample) const override;

    bool sampleDistance1(MediumSampleRecord *mRec,
                         const Ray &ray,
                         const Intersection &its,
                         Point2d sample) const;

    bool sampleDistance2(MediumSampleRecord *mRec,
                         const Ray &ray,
                         const Intersection &its,
                         Point2d sample) const;

    virtual Spectrum evalTransmittance(Point3d from,
                                       Point3d dest) const override;

    virtual Spectrum evalTransmittance2(Point3d from,
                                        Point3d dest,
                                        MediumState *mediumState) const override;
    Spectrum evalTransmittanceOpt(Point3d from,
                               Point3d dest,
                               MediumState *meidumState) const;

    bool intersectGP(const Ray &ray, GPRealization &gpRealization, double &t, Sampler &sampler) const;

    bool intersectMean(const Ray &ray, double &t) const;

    virtual bool isGPIS() override { return true; }

    std::shared_ptr<GaussianProcess> getGP() const {
        return gaussianProcess;
    }

private:
    std::shared_ptr<GaussianProcess> gaussianProcess;

    int marchingNumSamplePoints;
    double marchingStepSize;
    double marchingDesiredCov;

    MemoryModel memoryModel = MemoryModel::RenewalPlus;
};