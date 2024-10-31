#pragma once
#include "VolPathIntegrator.h"
class VolPathIntegratorGPIS : public VolPathIntegrator {
public:
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
        int sampleCount = 0.;
        double sampleDistanceSuccessProb = 0.;

        //for debugging
        std::vector<double> ts;
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

protected:
    const double trainingSPPFraction = .25;
    const int nPathLengthLimit = 64;

    std::mutex mutex;
};