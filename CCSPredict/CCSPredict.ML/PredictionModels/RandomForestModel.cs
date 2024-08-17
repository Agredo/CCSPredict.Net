using CCSPredict.Data;
using Microsoft.ML;

namespace CCSPredict.ML.PredictionModels;

public class RandomForestModel : PredictionModel
{
    public RandomForestModel(ICcsDataProvider dataProvider) : base(dataProvider)
    {
    }
    public async Task TrainAsync()
    {
        await base.TrainAsync();

        var pipeline = mlContext.Transforms.CopyColumns("Label", "CcsValue")
            .Append(mlContext.Transforms.Concatenate("Features", GetFeatureColumnNames()))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.FastForest(
                labelColumnName: "CcsValue",
                numberOfTrees: 100,
                numberOfLeaves: 20,
                minimumExampleCountPerLeaf: 10));

        model = await Task.Run(() => pipeline.Fit(data));
    }
}
