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
        }
        gpRealization.applyMemoryModel(r.direction, memoryModel);
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
    const double eps = 1e-6;
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

    bool goodFPTSample = false, intersected = false;
    GaussianProcess conditionedGaussianProcess;
    if (gpRealization.isEmpty()) {
        gpRealization.gp = gaussianProcess.get();
        std::tie(goodFPTSample, intersected) = gaussianProcess->sampleFPT(r, t, sampler);
    } else {
        auto transientGlobalCondition = gaussianProcess->globalCondition;
        for (int i = 0; i < gpRealization.size(); ++i) {
            transientGlobalCondition.points.push_back(gpRealization.points[i]);
            transientGlobalCondition.derivativeTypes.push_back(gpRealization.derivativeTypes[i]);
            transientGlobalCondition.derivativeDirections.push_back(gpRealization.derivativeDirections[i]);
            transientGlobalCondition.values.push_back(gpRealization.values[i]);
        }
        conditionedGaussianProcess = GaussianProcess(gaussianProcess->meanFunction, gaussianProcess->covFunction, transientGlobalCondition);
        std::tie(goodFPTSample, intersected) = conditionedGaussianProcess.sampleFPT(r, t, sampler);
    }

    if (!goodFPTSample) {
        intersected = false;
        t = ray.timeMin;
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
            }
            gpRealization.applyMemoryModel(r.direction, memoryModel);
        } while (!intersected && r.timeMax - t > eps);
        return intersected;
    } else {
        if (!intersected) return false;
        Point3d intersection = r.origin + r.direction * t;
        mRec->marchLength = t;
        mRec->scatterPoint = intersection;

        double lambda2 = 0.;
        if (conditionedGaussianProcess) {
            lambda2 = conditionedGaussianProcess.covSym(&intersection, DerivativeTypeFirst(), nullptr, 1, ray.direction)(0, 0);
        } else {
            lambda2 = gaussianProcess->covSym(&intersection, DerivativeTypeFirst(), nullptr, 1, ray.direction)(0, 0);
        }
        double Z = lambda2 * fm::sqrt(-2 * std::log(sampler.sample1D()));

        gpRealization.manualIntersectionAndNormal(intersection, r.direction, Z + gaussianProcess->mean(&intersection, DerivativeTypeFirst(), nullptr, 1, ray.direction)(0));
        mRec->aniso = normalize(gpRealization.sampleGradient(intersection, r.direction, sampler));
        gpRealization.applyMemoryModel(r.direction, memoryModel);
        return true;
    }
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
#else
    double eps = 1e-4;
    Vec3d direction = (from - dest);
    if (direction.length() < 1e-4) {
        return 1.;
    }
    direction = normalize(direction);
    Ray ray{dest, direction};
    ray.timeMin = eps;
    Intersection its;
    its.t = (from - dest).length() - eps;

    MediumSampleRecord sampleRecord{};
    sampleRecord.mediumState = mediumState;
    bool shadowed = sampleDistance(&sampleRecord, ray, its, {});

    return 1. - shadowed;
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
