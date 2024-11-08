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
#if USE_TRAINING_SAMPLES
                film->deposit(pixelPosition, L);
#endif

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
            /*{
                std::lock_guard guard(mutex);
                for (double grad : optInfo.graidents) {
                    std::cout << grad << " ";
                }
                if (optInfo.graidents.size()) {
                    std::cout << '\n';
                    std::fflush(stdout);
                }
            }*/

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
                        double grad = dot(mRec.aniso, ray.direction);
                        optInfo.gradientSum += grad;
                        optInfo.squaredGraidentSum += grad * grad;
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

                            Vec2d sample = rand_normal_2(*sampler);
                            mRec.marchLength = mean + sigma * sample[0];
                            mRec.scatterPoint = ray.origin + ray.direction * mRec.marchLength;

#if (GPIS_SAMPLE_NORMAL_METHOD == 0)
                            //we have statistic of ray direction gradient
                            double rayGradMean = optInfo.gradientSum / optInfo.sampleCount;
                            double rayGradSigma = fm::sqrt(optInfo.squaredGraidentSum / optInfo.sampleCount - rayGradMean * rayGradMean);
                            double sampleRayDirGrad = rayGradMean + sample[1] * rayGradSigma;

                            std::shared_ptr<GaussianProcess> gp = static_cast<GPISMedium *>(medium.get())->getGP();
                            Point3d intersection = mRec.scatterPoint;

                            std::vector<Point3d> points = {intersection, intersection};
                            std::vector<double> values = {0., sampleRayDirGrad};
                            std::vector<DerivativeType> derivativeTypes = {DerivativeType::None, DerivativeType::First};

                            mediumState.realization = GPRealization(gp.get(), points.data(), derivativeTypes.data(), nullptr, values.data(), points.size(), ray.direction);
                            mediumState.realization.justIntersected = true;
                            mRec.aniso = normalize(mediumState.realization.sampleGradient(intersection, ray.direction, *sampler));

                            mediumState.realization.applyMemoryModel(ray.direction, MemoryModel::Renewal);
#elif (GPIS_SAMPLE_NORMAL_METHOD == 1)
                            Frame frame(ray.direction);
                            mRec.aniso = normalize(frame.toWorld({gp->meanFunction->operator()(DerivativeType::First, pseudoPoint, frame.s),
                                                                  gp->meanFunction->operator()(DerivativeType::First, pseudoPoint, frame.t),
                                                                  gp->meanFunction->operator()(DerivativeType::First, pseudoPoint, frame.n)}));
#endif
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
                PathIntegratorLocalRecord sampleLightRecord = sampleDirectLightingGPISOpt(scene, mediumScatteringPoint, ray, &mediumState);
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
            auto [sampleIts, tr] = intersectIgnoreSurfaceGPISOpt(scene, ray, medium, &mediumState);
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
                PathIntegratorLocalRecord sampleLightRecord = sampleDirectLightingGPISOpt(scene, its, ray, &mediumState);
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

            auto [sampleIts, tr] = intersectIgnoreSurfaceGPISOpt(scene, ray, medium, &mediumState);

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

Spectrum VolPathIntegratorGPIS::evalTransmittanceGPISOpt(std::shared_ptr<Scene> scene, const Intersection &its, Point3d pointOnLight, const MediumState *mediumState) const {
#if ENABLE_GPIS_VISIBILITY_OPTIMIZATION
    MediumState transientMeidumState = *mediumState;
    float tmax = (pointOnLight - its.position).length();
    Ray shadowRay{its.position, normalize(pointOnLight - its.position), 1e-4f, tmax - 1e-4f};
    std::shared_ptr<Medium> medium = its.medium;
    Spectrum tr(1.f);
    while (true) {
        auto itsOpt = scene->intersect(shadowRay);

        if (medium) {
            if (!itsOpt) {
                if (!medium->isGPIS()) {
                    tr *= medium->evalTransmittance2(shadowRay.origin, pointOnLight, &transientMeidumState);
                } else {
                    tr *= static_cast<GPISMedium *>(medium.get())->evalTransmittanceMean(shadowRay.origin, pointOnLight, &transientMeidumState);
                }
                break;
            }

            if (!itsOpt->material->getBxDF(*itsOpt)->isNull()) {
                tr = .0f;
                break;
            }

            if (!medium->isGPIS()) {
                tr *= medium->evalTransmittance2(shadowRay.origin, pointOnLight, &transientMeidumState);
            } else {
                tr *= static_cast<GPISMedium *>(medium.get())->evalTransmittanceMean(shadowRay.origin, pointOnLight, &transientMeidumState);
            }
            medium = getTargetMedium(*itsOpt, shadowRay.direction);
            transientMeidumState.reset();
            shadowRay.origin = itsOpt->position;
            shadowRay.timeMax -= itsOpt->t;
        } else {
            if (!itsOpt) break;

            if (!itsOpt->material->getBxDF(*itsOpt)->isNull()) {
                tr = .0f;
                break;
            }
            medium = getTargetMedium(*itsOpt, shadowRay.direction);
            transientMeidumState.reset();
            shadowRay.origin = itsOpt->position;
            shadowRay.timeMax -= itsOpt->t;
        }
    }

    return tr;
#else
    return evalTransmittance2(scene, its, pointOnLight, mediumState);
#endif
}

PathIntegratorLocalRecord VolPathIntegratorGPIS::sampleDirectLightingGPISOpt(std::shared_ptr<Scene> scene, const Intersection &its, const Ray &ray, const MediumState *mediumState) {
#if ENABLE_GPIS_VISIBILITY_OPTIMIZATION
    auto [light, pdfChooseLight] = chooseOneLight(scene, sampler->sample1D());
    auto record = light->sampleDirect(its, sampler->sample2D(), ray.timeMin);
    double pdfDirect = record.pdfDirect * pdfChooseLight;// pdfScatter with respect to solid angle
    Vec3d dirScatter = record.wi;
    Point3d posL = record.dst;
    Point3d posS = its.position;
    auto transmittance = evalTransmittanceGPISOpt(scene, its, record.dst, mediumState);
    //    if (!its.material && transmittance.sum() < 2.9f) {
    //        std::cout << transmittance.sum() << "\n";
    //    }
    return {dirScatter, transmittance * record.s, pdfDirect, record.isDeltaPos};
#else
    return sampleDirectLighting2(scene, its, ray, mediumState);
#endif
}

std::pair<std::optional<Intersection>, Spectrum> VolPathIntegratorGPIS::intersectIgnoreSurfaceGPISOpt(std::shared_ptr<Scene> scene, const Ray &ray, std::shared_ptr<Medium> medium, const MediumState *mediumState) const {
#if ENABLE_GPIS_VISIBILITY_OPTIMIZATION
    MediumState transientMeidumState = *mediumState;

    const double eps = 1e-5;
    Vec3d dir = ray.direction;

    Spectrum tr(1.0);
    Ray marchRay{ray.origin + dir * eps, dir};
    std::shared_ptr<Medium> currentMedium = medium;

    Point3d lastScatteringPoint = ray.origin;
    auto testRayItsOpt = scene->intersect(marchRay);

    // calculate the transmittance of last segment from lastScatteringPoint to testRayItsOpt.
    while (true) {

        // corner case: infinite medium or infinite light source.
        if (!testRayItsOpt.has_value()) {
            if (currentMedium != nullptr)
                tr = Spectrum(0.0);
            return {testRayItsOpt, tr};
        }

        auto testRayIts = testRayItsOpt.value();

        // corner case: non-null surface
        if (testRayIts.material != nullptr) {
            if (!testRayIts.material->getBxDF(testRayIts)->isNull()) {
                if (currentMedium != nullptr) {
                    if (!currentMedium->isGPIS()) {
                        tr *= currentMedium->evalTransmittance2(testRayIts.position, lastScatteringPoint, &transientMeidumState);
                    } else {
                        tr *= static_cast<GPISMedium *>(currentMedium.get())->evalTransmittanceMean(testRayIts.position, lastScatteringPoint, &transientMeidumState);
                    }
                }
                return {testRayItsOpt, tr};
            }
        }

        // hit a null surface, calculate tr
        if (currentMedium != nullptr) {
            if (currentMedium != nullptr) {
                if (!currentMedium->isGPIS()) {
                    tr *= currentMedium->evalTransmittance2(testRayIts.position, lastScatteringPoint, &transientMeidumState);
                } else {
                    tr *= static_cast<GPISMedium *>(currentMedium.get())->evalTransmittanceMean(testRayIts.position, lastScatteringPoint, &transientMeidumState);
                }
            }
        }

        // update medium
        currentMedium = getTargetMedium(testRayIts, dir);
        transientMeidumState.reset();

        // update ray and intersection point.
        marchRay.origin = testRayIts.position + dir * eps;
        lastScatteringPoint = testRayIts.position;
        testRayItsOpt = scene->intersect(marchRay);
    }
    return {testRayItsOpt, tr};
#else
    return intersectIgnoreSurface2(scene, ray, medium, mediumState);
#endif
}
