using CCSPredict.Data;
using Microsoft.ML;

namespace CCSPredict.ML.PredictionModels;

public class NeuralNetworkModel : PredictionModel
{
    public NeuralNetworkModel(ICcsDataProvider dataProvider) : base(dataProvider)
    {
    }
    public async Task TrainAsync()
    {
        await base.TrainAsync();

        var pipeline = mlContext.Transforms.CopyColumns("Label", "CcsValue")
            .Append(mlContext.Transforms.Concatenate("Features", GetFeatureColumnNames()))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(
                labelColumnName: "Label",
                featureColumnName: "Features",
                l2Regularization: 0.01f,
                l1Regularization: 0.01f,
                optimizationTolerance: 1e-07f,
                historySize: 50));

        model = await Task.Run(() => pipeline.Fit(data));
    }
}
