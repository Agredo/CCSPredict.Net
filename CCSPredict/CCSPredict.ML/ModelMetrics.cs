namespace CCSPredict.ML;

public class ModelMetrics
{
    public double MeanAbsoluteError { get; internal set; }
    public double RSquared { get; internal set; }
    public double MeanSquaredError { get; internal set; }
    public double RootMeanSquaredError { get; internal set; }
}
