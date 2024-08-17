using CCSPredict.Data;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Trainers;
using static Microsoft.ML.DataOperationsCatalog;

namespace CCSPredict.ML.PredictionModels;

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

    public async Task OptimizeParametersAsync()
    {
        await base.TrainAsync();

        var experimentSettings = new RegressionExperimentSettings
        {
            OptimizingMetric = RegressionMetric.RSquared,
            CacheBeforeTrainer = CacheBeforeTrainer.On,
            MaxModels = 100
        };

        TrainTestData trainValidationData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        SweepablePipeline pipeline =
            mlContext.Auto().Featurizer(data)
                .Append(mlContext.Auto().Regression(labelColumnName: "CcsValue"));

        var experiment = mlContext.Auto().CreateExperiment();

        experiment.SetPipeline(pipeline)
            .SetRegressionMetric(RegressionMetric.RSquared, labelColumn: "CcsValue")
            .SetDataset(trainValidationData);

        TrialResult experimentResults = await experiment.RunAsync();

        model = experimentResults.Model;

    }
}
