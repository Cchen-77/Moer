#pragma once
#include "VolPathIntegrator.h"
#define ENABLE_GPIS_VISIBILITY_OPTIMIZATION 1

// 0: sample normal from GP with only zero-crossing conditioning
// 1: directly use the mean normal as gpis normal

#define GPIS_SAMPLE_NORMAL_METHOD 0

// whether to include the contribution of training samples in the final result
#define USE_TRAINING_SAMPLES 1

class VolPathIntegratorGPIS : public VolPathIntegrator {
public:
    // ONE: training the GPIS intersection range with few spp and shrincking ray's [tMin,tMax] acrodingly.
    // helping capturing the small details better but doing nothing to efficiency improvement
    //
    // TWO: trainng the distribution of GPIS intersection and we can get intersection from distribution directly.
    // Since we only have intersection's distribution,we need a way to get the corresponding normal

    // all optimizations are only for camera ray - GPIS intersection
    enum class OptimizingMethod {
        ONE,
        TWO,
        THREE
    };
    // some info needed for optimization
    struct GPISOptimzationInfo {
        enum class InfoState {
            TRAINING,
            OPTIMIZING,
            DISABLE
        } state = InfoState::DISABLE;
        // for method 1
        double tMin = std::numeric_limits<double>::max();
        double tMax = 0.;

        // for method 2
        double tSum = 0.;
        double squaredTSum = 0.;
        double gradientSum = 0;
        double squaredGraidentSum = 0.;
        int sampleCount = 0.;
        double sampleDistanceSuccessProb = 0.;

        // for debugging
        std::vector<double> ts;
        std::vector<double> graidents;
    };
    VolPathIntegratorGPIS(std::shared_ptr<Camera> _camera,
                          std::unique_ptr<Film> _film,
                          std::unique_ptr<TileGenerator> _tileGenerator,
                          std::shared_ptr<Sampler> _sampler,
                          int _spp,
                          int _renderThreadNum = 4);
    void render(std::shared_ptr<Scene> scene) override;

    void renderPerThread(const std::shared_ptr<Scene> &scene, OptimizingMethod method);

    Spectrum LiTraining(const Ray &ray,
                        std::shared_ptr<Scene> scene,
                        GPISOptimzationInfo &optInfo,
                        OptimizingMethod method);

    Spectrum LiOptimized(const Ray &ray,
                         std::shared_ptr<Scene> scene,
                         const GPISOptimzationInfo &optInfo,
                         OptimizingMethod method);

    Spectrum evalTransmittanceGPISOpt(std::shared_ptr<Scene> scene,
                                      const Intersection &its,
                                      Point3d pointOnLight,
                                      const MediumState *mediumState) const;

    PathIntegratorLocalRecord sampleDirectLightingGPISOpt(std::shared_ptr<Scene> scene,
                                                          const Intersection &its,
                                                          const Ray &ray,
                                                          const MediumState *mediumState);

    std::pair<std::optional<Intersection>, Spectrum>
    intersectIgnoreSurfaceGPISOpt(std::shared_ptr<Scene> scene,
                                  const Ray &ray,
                                  std::shared_ptr<Medium> medium,
                                  const MediumState *mediumState) const;

protected:
    const double trainingSPPFraction = 0.05;
    const int nPathLengthLimit = 1;

    std::mutex mutex;
};