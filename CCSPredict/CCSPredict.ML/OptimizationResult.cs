using Microsoft.ML;

namespace CCSPredict.ML;

public class OptimizationResult
{
    public IEstimator<ITransformer> BestPipeline { get; set; }
    public ITransformer BestModel { get; set; }
    public ModelMetrics Metrics { get; set; }
}