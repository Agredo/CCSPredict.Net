using CCSPredict.Data;
using Microsoft.ML;

namespace CCSPredict.ML;

public class SvmModel : PredictionModel
{
    public SvmModel(ICcsDataProvider dataProvider) : base(dataProvider)
    {
        
    }

    public async Task TrainAsync()
    {
        await base.TrainAsync();

        var pipeline = mlContext.Transforms.CopyColumns("Label", "CcsValue")
            .Append(mlContext.Transforms.Concatenate("Features", GetFeatureColumnNames()))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "CcsValue", maximumNumberOfIterations: 100));

        model = await Task.Run(() => pipeline.Fit(data));
    }
}
