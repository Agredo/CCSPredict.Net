using CCSPredict.Data;
using CCSPredict.Descriptors;
using Microsoft.ML;

namespace CCSPredict.ML.PredictionModels;

public class FastTreePredictionModel : PredictionModel
{
    public FastTreePredictionModel(CombinedFloatDescriptorCalculator descriptorCalculator, CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator, IEnumerable<MoleculeData> moleculeData) 
        : base(descriptorCalculator, bitVectorDescriptorCalculator, moleculeData)
    {
    }

    public async Task TrainAsync(double testFraction)
    {
        await base.TrainAsync(testFraction);

        var pipeline = mlContext.Transforms.CopyColumns("Label", "CcsValue")
            .Append(mlContext.Transforms.Concatenate("Features", GetFeatureColumnNames()))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "CcsValue"));

        model = pipeline.Fit(traingData);
    }
}
