using CCSPredict.Data;
using CCSPredict.Descriptors;
using Microsoft.ML;

namespace CCSPredict.ML.PredictionModels;

public class NeuralNetworkModel : PredictionModel
{
    public NeuralNetworkModel(CombinedFloatDescriptorCalculator descriptorCalculator, CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator, IEnumerable<MoleculeData> moleculeData)
        : base(descriptorCalculator, bitVectorDescriptorCalculator, moleculeData)
    {
    }
    public async Task TrainAsync(double testFraction)
    {
        await base.TrainAsync(testFraction);

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

        model = await Task.Run(() => pipeline.Fit(traingData));
    }
}
