using CCSPredict.Data;
using Microsoft.ML;

namespace CCSPredict.ML;

public class FastTreePredictionModel : PredictionModel
{
    public FastTreePredictionModel(ICcsDataProvider dataProvider) : base(dataProvider)
    {
    }

    public async Task TrainAsync()
    {
        await base.TrainAsync();

        var pipeline = mlContext.Transforms.CopyColumns("Label", "CcsValue")
            .Append(mlContext.Transforms.Concatenate("Features", GetFeatureColumnNames()))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "CcsValue"));

        model = pipeline.Fit(data);
    }
}
