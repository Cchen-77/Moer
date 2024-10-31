#include "VolPathIntegrator-gpis.h"

#include <mutex>
#include "CoreLayer/Math/Warp.h"
#include "FastMath.h"
#include "FunctionLayer/Material/NullMaterial.h"

// Optimization Method 2 need
#include "FunctionLayer/GaussianProcess/GaussianProcessUtils.h"
#include "FunctionLayer/GaussianProcess/GaussianProcess.h"
#include "FunctionLayer/Medium/GPISMedium.h"

VolPathIntegratorGPIS::VolPathIntegratorGPIS(std::shared_ptr<Camera> _camera,
                                             std::unique_ptr<Film> _film,
                                             std::unique_ptr<TileGenerator> _tileGenerator,
                                             std::shared_ptr<Sampler> _sampler,
                                             int _spp, int _renderThreadNum)
    : VolPathIntegrator(std::move(_camera), std::move(_film), std::move(_tileGenerator), std::move(_sampler), _spp, _renderThreadNum) {
}

void VolPathIntegratorGPIS::render(std::shared_ptr<Scene> scene) {
    std::vector<std::thread> threads;
    for (int i = 0; i < renderThreadNum; i++) {
        threads.push_back(std::thread(&VolPathIntegratorGPIS::renderPerThread, this, scene, OptimizingMethod::TWO));
    }

    for (int i = 0; i < renderThreadNum; i++) {
        threads[i].join();
    }

    printProgress(1.f);
}

void VolPathIntegratorGPIS::renderPerThread(const std::shared_ptr<Scene> &scene, OptimizingMethod method) {
    /**
     * @warning Other part of Integrator uses the original sampler
     *          so I have to fill its vectors here.
     *          In fact every time a fresh sampler is needed we should
     *          use Sampler::clone() to get one.
     */
    static int tileFinished = 0;

    sampler->startPixel({0, 0});
    auto ssampler = sampler->clone(0);
    while (true) {
        auto optionalTile = tileGenerator->generateNextTile();
        if (optionalTile == std::nullopt)
            break;
        auto tile = optionalTile.value();

        for (auto it = tile->begin(); it != tile->end(); ++it) {
            auto pixelPosition = *it;

            const auto &cam = *this->camera;
            /**
             * @bug Sampler is NOT designed for multi-threads, need copy for each thread.
             *      Sampler::clone() will return a Sampler copy, only with same sampling
             *      strategy, random numbers are not guaranteed to be identical.
             */
            // sampler->startPixel(pixelPosition);
            ssampler->startPixel(pixelPosition);

            GPISOptimzationInfo optInfo;
            optInfo.state = GPISOptimzationInfo::InfoState::TRAINING;

            // need at least one spp for training.If spp is '1',degenerate to normal volpathtracer
            int trainingSPP = std::max(1, (int)(spp * trainingSPPFraction));

            int optimizedSPP = spp - trainingSPP;
            for (int i = 0; i < trainingSPP; ++i) {
                auto L = LiTraining(
                    cam.generateRay(
                        film->getResolution(),
                        pixelPosition,
                        ssampler->getCameraSample()),
                    scene,
                    optInfo,
                    method);
                // don't waste training samples
                // film->deposit(pixelPosition, L);

                /**
                 * @warning spp used in this for loop belongs to Integrator.
                 *          It is irrelevant with spp passed to Sampler.
                 *          And Sampler has no sanity check for subscript
                 *          of sample vector. Error may occur if spp passed to
                 *          Integrator is bigger than which passed to Sampler.
                 */
                ssampler->nextSample();
            }

            // DEBUG
            {
                /* std::lock_guard guard(mutex);
                 for (double t : optInfo.ts) {
                     std::cout << t << " ";
                 }
                 if (optInfo.ts.size()) {
                     std::cout << '\n';
                     std::fflush(stdout);
                 }*/
            }

            optInfo.state = GPISOptimzationInfo::InfoState::OPTIMIZING;
            switch (method) {
                case OptimizingMethod::ONE:
                    break;
                case OptimizingMethod::TWO:
                    optInfo.sampleDistanceSuccessProb = 1. * optInfo.sampleCount / trainingSPP;
                    break;
                case OptimizingMethod::THREE:
                    break;
                default:
                    break;
            }

            for (int i = 0; i < optimizedSPP; ++i) {
                auto L = LiOptimized(
                    cam.generateRay(
                        film->getResolution(),
                        pixelPosition,
                        ssampler->getCameraSample()),
                    scene,
                    optInfo,
                    method);
                film->deposit(pixelPosition, L);
                /**
                 * @warning spp used in this for loop belongs to Integrator.
                 *          It is irrelevant with spp passed to Sampler.
                 *          And Sampler has no sanity check for subscript
                 *          of sample vector. Error may occur if spp passed to
                 *          Integrator is bigger than which passed to Sampler.
                 */
                ssampler->nextSample();
            }
        }

        //* Finish one tile rendering
        if (++tileFinished % 5) {
            printProgress((float)tileFinished / tileGenerator->tileCount);
        }
    }
}

Spectrum VolPathIntegratorGPIS::LiTraining(const Ray &initialRay, std::shared_ptr<Scene> scene, GPISOptimzationInfo &optInfo, OptimizingMethod method) {
    Spectrum L{.0};
    Spectrum throughput{1.0};

    Ray ray = initialRay;

    const double eps = 1e-5;
    int nBounces = 0;
    bool specularBounce = false;
    PathIntegratorLocalRecord prevLightSampleRecord;

    std::shared_ptr<Medium> medium = nullptr;
    // mainly for GPIS medium now,but maybe some other medium need it also.
    MediumState mediumState{*sampler};

    auto itsOpt = scene->intersect(ray);

    while (true) {

        MediumSampleRecord mRec{};
        mRec.mediumState = &mediumState;
        if (medium && medium->sampleDistanceSafe(&mRec, ray, itsOpt, sampler->sample2D())) {

            // Handle medium distance sampling
            throughput *= mRec.tr * mRec.sigmaS / mRec.pdf;
            if (nBounces == 0 && medium->isGPIS() && optInfo.state == GPISOptimzationInfo::InfoState::TRAINING) {
                // optInfo.ts.push_back(mRec.marchLength);
                switch (method) {
                    case OptimizingMethod::ONE: {
                        double t = mRec.marchLength;
                        optInfo.tMax = std::max(optInfo.tMax, t + 0.1);
                        optInfo.tMin = std::min(optInfo.tMin, t - 0.1);
                        break;
                    }
                    case OptimizingMethod::TWO: {
                        double t = mRec.marchLength;
                        optInfo.sampleCount++;
                        optInfo.tSum += t;
                        optInfo.squaredTSum += t * t;
                        break;
                    }
                    case OptimizingMethod::THREE: {
                        break;
                    }
                }
            }

            Intersection mediumScatteringPoint;
            if (!mRec.needAniso) {
                mediumScatteringPoint = fulfillScatteringPoint(mRec.scatterPoint, ray.direction, medium);
            } else {
                // abuse slightly for gpis medium
                mediumScatteringPoint = fulfillScatteringPoint(mRec.scatterPoint, mRec.aniso, medium);
                mediumScatteringPoint.geometryNormal = mRec.aniso;
            }
            //* ----- Luminaire Sampling -----
            for (int i = 0; i < nDirectLightSamples; ++i) {
                PathIntegratorLocalRecord sampleLightRecord = sampleDirectLighting2(scene, mediumScatteringPoint, ray, &mediumState);
                PathIntegratorLocalRecord evalScatterRecord = evalScatter(mediumScatteringPoint, ray, sampleLightRecord.wi);
                if (!sampleLightRecord.f.isBlack()) {
                    double misw = MISWeight(sampleLightRecord.pdf, evalScatterRecord.pdf);
                    if (sampleLightRecord.isDelta)
                        misw = 1.0;
                    L += throughput * sampleLightRecord.f * evalScatterRecord.f / sampleLightRecord.pdf * misw / nDirectLightSamples;
                }
            }

            //* ----- Phase Sampling -----

            PathIntegratorLocalRecord sampleScatterRecord = sampleScatter(mediumScatteringPoint, ray);
            if (sampleScatterRecord.f.isBlack())
                break;
            throughput *= sampleScatterRecord.f / sampleScatterRecord.pdf;
            ray = Ray{mediumScatteringPoint.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
            itsOpt = scene->intersect(ray);
            auto [sampleIts, tr] = intersectIgnoreSurface2(scene, ray, medium, &mediumState);
            auto evalLightRecord = evalEmittance(scene, sampleIts, ray);
            if (!evalLightRecord.f.isBlack()) {
                double misw = MISWeight(sampleScatterRecord.pdf, evalLightRecord.pdf);
                if (sampleScatterRecord.isDelta)
                    misw = 1.0;
                L += throughput * tr * evalLightRecord.f * misw;
            }
            nBounces++;
            if (nBounces > nPathLengthLimit || !itsOpt)
                break;
        } else {
            if (medium) throughput *= mRec.tr / mRec.pdf;

            {
                PathIntegratorLocalRecord evalLightRecord = evalEmittance(scene, itsOpt, ray);
                if (nBounces == 0) {
                    L += throughput * evalLightRecord.f;
                }
            }
            // there will be case that itsOpt is null when light cross null interfaces and hit nothing,
            // for that case we still need to calculate evnlight for it.
            if (!itsOpt) break;

            auto its = itsOpt.value();
            its.medium = medium;

            if (its.material->getBxDF(its)->isNull()) {
                medium = getTargetMedium(its, ray.direction);
                mediumState.reset();
                ray = Ray{its.position + eps * ray.direction, ray.direction};
                itsOpt = scene->intersect(ray);
                continue;
            }

            //* Direct Illumination
            for (int i = 0; i < nDirectLightSamples; ++i) {
                PathIntegratorLocalRecord sampleLightRecord = sampleDirectLighting2(scene, its, ray, &mediumState);
                PathIntegratorLocalRecord evalScatterRecord = evalScatter(its, ray, sampleLightRecord.wi);

                if (!sampleLightRecord.f.isBlack()) {
                    //* Multiple importance sampling
                    double misw = MISWeight(sampleLightRecord.pdf, evalScatterRecord.pdf);
                    if (sampleLightRecord.isDelta)
                        misw = 1.0;
                    L += throughput * sampleLightRecord.f * evalScatterRecord.f / sampleLightRecord.pdf * misw / nDirectLightSamples;
                }
            }

            //* ----- BSDF Sampling -----
            PathIntegratorLocalRecord sampleScatterRecord = sampleScatter(its, ray);
            if (sampleScatterRecord.f.isBlack())
                break;
            throughput *= sampleScatterRecord.f / sampleScatterRecord.pdf;

            //* Test whether the sampling ray hit the emitter
            const double eps = 1e-4;
            ray = Ray{its.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
            itsOpt = scene->intersect(ray);

            auto [sampleIts, tr] = intersectIgnoreSurface2(scene, ray, medium, &mediumState);

            auto evalLightRecord = evalEmittance(scene, sampleIts, ray);
            if (!evalLightRecord.f.isBlack()) {
                //* The sampling ray hit the emitter
                //* Multiple importance sampling
                double misw = MISWeight(sampleScatterRecord.pdf, evalLightRecord.pdf);
                if (sampleScatterRecord.isDelta)
                    misw = 1.0;
                L += throughput * tr * evalLightRecord.f * misw;
            }
        }
        nBounces++;
        if (nBounces > nPathLengthLimit || !itsOpt) break;
        double pSurvive = russianRoulette(throughput, nBounces);
        if (randFloat() > pSurvive)
            break;
        throughput /= pSurvive;
    }

    return L;
}

Spectrum VolPathIntegratorGPIS::LiOptimized(const Ray &initialRay, std::shared_ptr<Scene> scene, const GPISOptimzationInfo &optInfo, OptimizingMethod method) {
    Spectrum L{.0};
    Spectrum throughput{1.0};

    Ray ray = initialRay;

    const double eps = 1e-5;
    int nBounces = 0;
    bool specularBounce = false;
    PathIntegratorLocalRecord prevLightSampleRecord;

    std::shared_ptr<Medium> medium = nullptr;
    // mainly for GPIS medium now,but maybe some other medium need it also.
    MediumState mediumState{*sampler};

    auto itsOpt = scene->intersect(ray);

    while (true) {
        bool sampleDistanceResult = false;
        MediumSampleRecord mRec{};
        mRec.mediumState = &mediumState;
        if (medium) {
            if (nBounces == 0 && medium->isGPIS() && optInfo.state == GPISOptimzationInfo::InfoState::OPTIMIZING) {
                switch (method) {
                    case OptimizingMethod::ONE: {
                        if (optInfo.tMin > optInfo.tMax) {
                            mRec.needAniso = true;
                            mRec.tr = 1.;
                            mRec.pdf = 1.;
                            mRec.sigmaS = 1.;
                            mRec.sigmaA = 0.;
                        } else {
                            Ray r = ray;
                            r.timeMax = std::min(r.timeMax, optInfo.tMax);
                            r.timeMin = std::max(r.timeMin, optInfo.tMin);
                            sampleDistanceResult = medium->sampleDistanceSafe(&mRec, r, itsOpt, sampler->sample2D());
                        }
                        break;
                    }
                    case OptimizingMethod::TWO: {
                        mRec.tr = 1.;
                        mRec.sigmaS = 1.;
                        mRec.pdf = 1.;
                        mRec.sigmaA = 0.;
                        mRec.needAniso = true;
                        if (randFloat() < optInfo.sampleDistanceSuccessProb) {
                            double mean = optInfo.tSum / optInfo.sampleCount;
                            double sigma = fm::sqrt(optInfo.squaredTSum / optInfo.sampleCount - mean * mean);
                            // kind of waste,we just need one sample actually
                            double sample = rand_normal_2(*sampler)[0];
                            sample = mean + sigma * sample;

                            mRec.marchLength = sample;
                            mRec.scatterPoint = ray.origin + ray.direction * sample;

                            // need a extra pseudo marching point "mRec.scatterPoint - 0.01*ray.direction" to ensure normal outward
                            Point3d intersection = mRec.scatterPoint;
                            Point3d pseudoPoint = mRec.scatterPoint - 0.1 * ray.direction;
                            std::shared_ptr<GaussianProcess> gp = static_cast<GPISMedium *>(medium.get())->getGP();
                            std::vector<Point3d> points = {intersection, pseudoPoint};
                            std::vector<DerivativeType> derivativeTypes = {DerivativeType::None, DerivativeType::None};
                            std::vector<double> values = {0, gp->meanFunction->operator()(DerivativeType::None, pseudoPoint)};

                            mediumState.realization = GPRealization(gp.get(), points.data(), derivativeTypes.data(), nullptr, values.data(), 1, {});
                            Frame frame(ray.direction);
                            mRec.aniso = normalize(frame.toWorld({gp->meanFunction->operator()(DerivativeType::First, pseudoPoint, frame.s),
                                                                  gp->meanFunction->operator()(DerivativeType::First, pseudoPoint, frame.t),
                                                                  gp->meanFunction->operator()(DerivativeType::First, pseudoPoint, frame.n)}));
                            // mediumState.realization.applyMemoryModel(ray.direction, MemoryModel::RenewalPlus);
                            sampleDistanceResult = true;
                        }
                        break;
                    }
                    case OptimizingMethod::THREE: {

                        break;
                    }
                }
            } else {
                sampleDistanceResult = medium->sampleDistanceSafe(&mRec, ray, itsOpt, sampler->sample2D());
            }
        }
        if (sampleDistanceResult) {
            // Handle medium distance sampling
            throughput *= mRec.tr * mRec.sigmaS / mRec.pdf;

            Intersection mediumScatteringPoint;
            if (!mRec.needAniso) {
                mediumScatteringPoint = fulfillScatteringPoint(mRec.scatterPoint, ray.direction, medium);
            } else {
                // abuse slightly for gpis medium
                mediumScatteringPoint = fulfillScatteringPoint(mRec.scatterPoint, mRec.aniso, medium);
                mediumScatteringPoint.geometryNormal = mRec.aniso;
            }
            //* ----- Luminaire Sampling -----
            for (int i = 0; i < nDirectLightSamples; ++i) {
                PathIntegratorLocalRecord sampleLightRecord = sampleDirectLighting2(scene, mediumScatteringPoint, ray, &mediumState);
                PathIntegratorLocalRecord evalScatterRecord = evalScatter(mediumScatteringPoint, ray, sampleLightRecord.wi);
                if (!sampleLightRecord.f.isBlack()) {
                    double misw = MISWeight(sampleLightRecord.pdf, evalScatterRecord.pdf);
                    if (sampleLightRecord.isDelta)
                        misw = 1.0;
                    L += throughput * sampleLightRecord.f * evalScatterRecord.f / sampleLightRecord.pdf * misw / nDirectLightSamples;
                }
            }

            //* ----- Phase Sampling -----

            PathIntegratorLocalRecord sampleScatterRecord = sampleScatter(mediumScatteringPoint, ray);
            if (sampleScatterRecord.f.isBlack())
                break;
            throughput *= sampleScatterRecord.f / sampleScatterRecord.pdf;
            ray = Ray{mediumScatteringPoint.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
            itsOpt = scene->intersect(ray);
            auto [sampleIts, tr] = intersectIgnoreSurface2(scene, ray, medium, &mediumState);
            auto evalLightRecord = evalEmittance(scene, sampleIts, ray);
            if (!evalLightRecord.f.isBlack()) {
                double misw = MISWeight(sampleScatterRecord.pdf, evalLightRecord.pdf);
                if (sampleScatterRecord.isDelta)
                    misw = 1.0;
                L += throughput * tr * evalLightRecord.f * misw;
            }
            nBounces++;
            if (nBounces > nPathLengthLimit || !itsOpt)
                break;
        } else {
            if (medium) throughput *= mRec.tr / mRec.pdf;

            {
                PathIntegratorLocalRecord evalLightRecord = evalEmittance(scene, itsOpt, ray);
                if (nBounces == 0) {
                    L += throughput * evalLightRecord.f;
                }
            }
            // there will be case that itsOpt is null when light cross null interfaces and hit nothing,
            // for that case we still need to calculate evnlight for it.
            if (!itsOpt) break;

            auto its = itsOpt.value();
            its.medium = medium;

            if (its.material->getBxDF(its)->isNull()) {
                medium = getTargetMedium(its, ray.direction);
                mediumState.reset();
                ray = Ray{its.position + eps * ray.direction, ray.direction};
                itsOpt = scene->intersect(ray);
                continue;
            }

            //* Direct Illumination
            for (int i = 0; i < nDirectLightSamples; ++i) {
                PathIntegratorLocalRecord sampleLightRecord = sampleDirectLighting2(scene, its, ray, &mediumState);
                PathIntegratorLocalRecord evalScatterRecord = evalScatter(its, ray, sampleLightRecord.wi);

                if (!sampleLightRecord.f.isBlack()) {
                    //* Multiple importance sampling
                    double misw = MISWeight(sampleLightRecord.pdf, evalScatterRecord.pdf);
                    if (sampleLightRecord.isDelta)
                        misw = 1.0;
                    L += throughput * sampleLightRecord.f * evalScatterRecord.f / sampleLightRecord.pdf * misw / nDirectLightSamples;
                }
            }

            //* ----- BSDF Sampling -----
            PathIntegratorLocalRecord sampleScatterRecord = sampleScatter(its, ray);
            if (sampleScatterRecord.f.isBlack())
                break;
            throughput *= sampleScatterRecord.f / sampleScatterRecord.pdf;

            //* Test whether the sampling ray hit the emitter
            const double eps = 1e-4;
            ray = Ray{its.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
            itsOpt = scene->intersect(ray);

            auto [sampleIts, tr] = intersectIgnoreSurface2(scene, ray, medium, &mediumState);

            auto evalLightRecord = evalEmittance(scene, sampleIts, ray);
            if (!evalLightRecord.f.isBlack()) {
                //* The sampling ray hit the emitter
                //* Multiple importance sampling
                double misw = MISWeight(sampleScatterRecord.pdf, evalLightRecord.pdf);
                if (sampleScatterRecord.isDelta)
                    misw = 1.0;
                L += throughput * tr * evalLightRecord.f * misw;
            }
        }
        nBounces++;
        if (nBounces > nPathLengthLimit || !itsOpt) break;
        double pSurvive = russianRoulette(throughput, nBounces);
        if (randFloat() > pSurvive)
            break;
        throughput /= pSurvive;
    }

    return L;
}
