#include "GPISMedium.h"
#include "GPISPhase.h"
#include "FunctionLayer/Sampler/Independent.h"
#include "FunctionLayer/GaussianProcess/GaussianProcessFactory.h"
#include "FunctionLayer/GaussianProcess/GaussianProcessUtils.h"

GPISMedium::GPISMedium(const Json &json) : Medium(std::make_shared<GPISPhase>(json["phase"])) {
    marchingNumSamplePoints = getOptional(json, "marching_num_sample_points", 8);
    if (marchingNumSamplePoints < 2) marchingNumSamplePoints = 2;
    marchingStepSize = getOptional(json, "marching_step_size", 0.);
    marchingDesiredCov = getOptional(json, "marching_desired_cov", 0.);

    gaussianProcess = GaussianProcessFactory::LoadGaussianProcessFromJson(json["gaussian_process"]);
}

bool GPISMedium::sampleDistance(MediumSampleRecord *mRec, const Ray &ray, const Intersection &its, Point2d sample) const {
#if defined(ENABLE_GPISMEDIUM)
#if (GPIS_LIGHT_TRANSPORT_VERSION == 1)
    const double eps = 1e-6;
    GPRealization &gpRealization = mRec->mediumState->realization;
    Sampler &sampler = mRec->mediumState->sampler;

    Ray r = ray;
    r.timeMax = std::min(r.timeMax, r.timeMin + 200);
    r.timeMax = std::min(its.t, r.timeMax);

    double t = r.timeMin;
    bool intersected = false;
    do {
        intersected = intersectGP(r, gpRealization, t, sampler);
        if (t < r.timeMax) {
            Point3d point = r.origin + t * r.direction;
            Vec3d grad = gpRealization.sampleGradient(point, r.direction, sampler);
            if (intersected) {
                mRec->aniso = normalize(grad);
                mRec->marchLength = t;
                mRec->scatterPoint = point;
            }
            gpRealization.applyMemoryModel(r.direction, memoryModel);
        }
    } while (!intersected && r.timeMax - t > eps);
    mRec->sigmaS = 1.;
    mRec->sigmaA = 0.;
    mRec->pdf = 1.;
    mRec->tr = 1.;
    mRec->needAniso = true;
    return intersected;
#elif (GPIS_LIGHT_TRANSPORT_VERSION == 2)
    // Performance optimized light transport which only work when Renewal or Renewal+ memory model applying
    // Gamma(t,n|zeta)
    // = kappa(n|t is first-passage-time,zeta) * first-passage-time-density(t|zeta)
    // = hat-kappa(n|t is a 0-downcrossing,zeta) * first-passage-density(t|zeta)
    Sampler &sampler = mRec->mediumState->sampler;
    GPRealization &gpRealization = mRec->mediumState->realization;

    Ray r = ray;
    r.timeMax = std::min(r.timeMax, r.timeMin + 200);
    r.timeMax = std::min(its.t, r.timeMax);

    double t = r.timeMin;

    mRec->sigmaS = 1.;
    mRec->sigmaA = 0.;
    mRec->needAniso = true;
    // since we sample from ff density directly,tr/pdf is always equals 1,so we don't need them
    mRec->tr = 1.;
    mRec->pdf = 1.;

    double Tr = 0.;
    if (gpRealization.isEmpty()) {
        gpRealization.gp = gaussianProcess.get();
        Tr = gaussianProcess->sampleFPT(r, t, marchingNumSamplePoints, sampler);
    } else {
        Tr = gaussianProcess->sampleFPTCond(r, t, marchingNumSamplePoints, sampler, EXPAND_GPREALIZATION_WITH_VALUE(gpRealization));
    }

    if (sampler.sample1D() < Tr) {
        return false;
    }

    Point3d intersection = r.origin + r.direction * t;

    mRec->marchLength = t;
    mRec->scatterPoint = intersection;
    gpRealization.manualIntersectionAndNormal(intersection, r.direction, -1);
    mRec->aniso = normalize(gpRealization.sampleGradient(intersection, r.direction, sampler));
    gpRealization.applyMemoryModel(ray.direction, MemoryModel::RenewalPlus);

    return true;
#else
    return false;
#endif
#else
    return false;
#endif
}

Spectrum GPISMedium::evalTransmittance(Point3d from, Point3d dest) const {
    // TODO(Cchen77):
    // limited by vol path tracer framework,we just have a naive version,which just like applying Renewal memory model before cast the shadowray
    // need a more elegant way
    IndependentSampler transientSampler(1, 5);// 1,5 is meaningless

    Vec3d direction = (dest - from);
    if (direction.isZero()) {
        return 1.;
    }
    direction = normalize(direction);
    Ray ray{from, direction};
    MediumState mediumState{transientSampler};
    mediumState.realization.reset();
    mediumState.realization.points.push_back(dest);
    mediumState.realization.values.push_back(0);
    mediumState.realization.derivativeTypes.push_back(DerivativeType::None);
    mediumState.realization.derivativeDirections.push_back({});

    MediumSampleRecord sampleRecord{};
    sampleRecord.mediumState = &mediumState;

    Intersection its;
    its.t = (dest - from)[0] / direction[0];
    bool shadowed = sampleDistance(&sampleRecord, ray, its, {});

    return 1 - shadowed;
}

Spectrum GPISMedium::evalTransmittance2(Point3d from, Point3d dest, MediumState *mediumState) const {
#if defined(ENABLE_GPISMEDIUM)
#if (GPIS_LIGHT_TRANSPORT_VERSION == 1)
    Vec3d direction = (dest - from);
    if (direction.length() < 1e-4) {
        return 1.;
    }
    direction = normalize(direction);
    Ray ray{from, direction};

    Intersection its;
    its.t = (dest - from).length();

    MediumSampleRecord sampleRecord{};
    sampleRecord.mediumState = mediumState;
    bool shadowed = sampleDistance(&sampleRecord, ray, its, {});

    return 1. - shadowed;
#elif (GPIS_LIGHT_TRANSPORT_VERSION == 2)
    Sampler &sampler = mediumState->sampler;
    GPRealization &gpRealization = mediumState->realization;

    Vec3d direction = (dest - from);
    if (direction.length() < 1e-4) {
        return 1.;
    }
    direction = normalize(direction);
    Ray ray{from, direction};
    ray.timeMax = (dest - from).length();

    double t = 0.;
    if (gpRealization.isEmpty()) {
        gpRealization.gp = gaussianProcess.get();
        return gaussianProcess->sampleFPT(ray, t, marchingNumSamplePoints, sampler);
    } else {
        return gaussianProcess->sampleFPTCond(ray, t, marchingNumSamplePoints, sampler, EXPAND_GPREALIZATION_WITH_VALUE(gpRealization));
    }
#else
    return 1.;
#endif

#else
    return 1.;
#endif
}

Spectrum GPISMedium::evalTransmittanceOpt(Point3d from, Point3d dest, MediumState *mediumState) const {

    Vec3d direction = (dest - from);
    if (direction.length() < 1e-4) {
        return 1.;
    }
    direction = normalize(direction);
    Ray ray{from, direction};

    ray.timeMax = std::min(ray.timeMax, ray.timeMin + 200);
    ray.timeMax = std::min((dest - from).length(), ray.timeMax);

    // bool shadowed = intersectMean(ray, t);

    double maxDistance = ray.timeMax;
    double determinedStepSize = maxDistance / (marchingNumSamplePoints - 1);

    if (marchingStepSize < determinedStepSize) {
        determinedStepSize = marchingStepSize;
    }
    int sampleCount = std::ceil(maxDistance / determinedStepSize);
    std::vector<Point3d> points;
    // nearby information is more important
    points.push_back(ray.origin + ray.direction * 0.001 * determinedStepSize);
    points.push_back(ray.origin + ray.direction * 0.01 * determinedStepSize);
    points.push_back(ray.origin + ray.direction * 0.02 * determinedStepSize);
    points.push_back(ray.origin + ray.direction * 0.04 * determinedStepSize);
    points.push_back(ray.origin + ray.direction * 0.08 * determinedStepSize);
    for (int i = 1; i <= sampleCount; ++i) {
        double t = (i == sampleCount) ? ray.timeMax : (i * determinedStepSize);
        Point3d p = ray.origin + t * ray.direction;
        points.push_back(p);
    }

    double UnshadowedPosibility1 = 1.;
    double UnshadowedPosibility2 = 1.;
    // we need to get covariance and mean from gp directly,utilize global conditioning feature to apply realization condition and get conditioned cov and mean.
    GaussianProcess transientGaussianProcess(gaussianProcess->meanFunction, gaussianProcess->covFunction, mediumState->realization);
    double eps = 1e-12;
    for (auto &point : points) {
        DerivativeType derivativeTypeNone = DerivativeType::None;
        auto [_mean, _cov] = transientGaussianProcess.meanAndCov(&point, &derivativeTypeNone, nullptr, 1, {});
        double mean = _mean(0);
        double variance = _cov(0, 0);
        if (variance < 1e-32) {
            // variance too small
            if (mean < 0.) {
                UnshadowedPosibility1 = 0.;
            }
            if (mean > 0.) {
                UnshadowedPosibility2 = 0.;
            }
        } else {
            double u = (0. - mean) / fm::sqrt(variance);
            double cdf = std::clamp(0.5 * std::erfc(-u / 1.4142135623730951), 0., 1.);
            UnshadowedPosibility1 *= 1. - cdf;
            UnshadowedPosibility2 *= cdf;
        }
    }
    return UnshadowedPosibility1 + UnshadowedPosibility2;

    /*bool havSample = false;
    double resSample = 0.;
    for (auto &point : points) {
        DerivativeType derivativeTypeNone = DerivativeType::None;
        auto [_mean, _cov] = transientGaussianProcess.meanAndCov( point, &derivativeTypeNone, nullptr, 1, {});

        double mean = _mean(0);
        double variance = _cov(0, 0);
        double sample;
        if (havSample) {
            sample = resSample;
            havSample = false;
        } else {
            auto sample2 = rand_normal_2(mediumState->sampler);
            sample = sample2[0];
            resSample = sample2[1];
            havSample = true;
        }
        double v = mean + sample * fm::sqrt(variance);
        if (v < 0.) {
            return 0.;
        }
    }
    return 1.;*/
}

bool GPISMedium::intersectGP(const Ray &ray, GPRealization &gpRealization, double &t, Sampler &sampler) const {
    double maxDistance = ray.timeMax - t;
    double determinedStepSize = maxDistance / (marchingNumSamplePoints - 1);

    if (marchingStepSize < determinedStepSize) {
        determinedStepSize = marchingStepSize;
    }
    determinedStepSize = std::min(determinedStepSize, gaussianProcess->goodStepSize(ray.origin + ray.direction * t, ray.direction, marchingDesiredCov, determinedStepSize));

    std::vector<Point3d> points;
    std::vector<DerivativeType> derivativeTypes;
    std::vector<double> ts;

    Point3d p = ray.origin + ray.direction * (determinedStepSize * 0.01 + t);
    points.push_back(p);
    derivativeTypes.push_back(DerivativeType::None);
    ts.push_back(determinedStepSize * 0.01 + t);

    for (int i = 1; i < marchingNumSamplePoints; ++i) {
        Point3d p = ray.origin + ray.direction * (i * determinedStepSize + t);
        points.push_back(p);
        derivativeTypes.push_back(DerivativeType::None);
        ts.push_back(i * determinedStepSize + t);
    }

    // it's the first time we have a realization,so we can sampling without condition
    if (gpRealization.isEmpty()) {
        gpRealization = gaussianProcess->sample(points.data(), derivativeTypes.data(), nullptr, marchingNumSamplePoints, {}, sampler);
    }
    // use last realization as conditon
    else {
        gpRealization = gaussianProcess->sampleCond(
            points.data(), derivativeTypes.data(), nullptr, marchingNumSamplePoints, {}, EXPAND_GPREALIZATION_WITH_VALUE(gpRealization), sampler);
    }
    double lastV = gpRealization.values[0];
    double lastT = ts[0];
    t = ts[0];
    for (int i = 1; i < marchingNumSamplePoints; ++i) {
        double curV = gpRealization.values[i];
        double curT = ts[i];
        if (curV * lastV < 0) {
            double offset = lastV / (lastV - curV);
            gpRealization.makeIntersection(i, offset);
            t = lerp(lastT, curT, offset);
            return true;
        }
        t = ts[i];
        lastV = curV;
        lastT = curT;
    }
    return false;
}

bool GPISMedium::intersectMean(const Ray &ray, double &t) const {
    double maxDistance = ray.timeMax - t;
    double determinedStepSize = maxDistance / (marchingNumSamplePoints - 1);

    if (marchingStepSize < determinedStepSize) {
        determinedStepSize = marchingStepSize;
    }
    determinedStepSize = std::min(determinedStepSize, gaussianProcess->goodStepSize(ray.origin + ray.direction * t, ray.direction, marchingDesiredCov, determinedStepSize));

    int sampleCount = std::ceil(maxDistance / determinedStepSize);

    double lastV = gaussianProcess->meanFunction->operator()(DerivativeType::None, ray.origin);
    double lastT = t;
    for (int i = 1; i <= sampleCount; ++i) {
        double curT = (i == sampleCount) ? ray.timeMax : (t + i * determinedStepSize);
        double curV = gaussianProcess->meanFunction->operator()(DerivativeType::None, ray.origin + curT * ray.direction);
        if (lastV * curV < 0.) {
            double offset = lastV / (lastV - curV);
            t = lerp(lastT, curT, offset);
            return true;
        }
        t = curT;
        lastV = curV;
        lastT = curT;
    }
    return false;
}
