using CCSPredict.Data;
using CCSPredict.Descriptors;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Trainers;
using static Microsoft.ML.DataOperationsCatalog;

namespace CCSPredict.ML.PredictionModels;

public class SvmModel : PredictionModel
{
    public SvmModel(CombinedFloatDescriptorCalculator descriptorCalculator, CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator, IEnumerable<MoleculeData> moleculeData)
        : base(descriptorCalculator, bitVectorDescriptorCalculator, moleculeData)
    {

    }
    public override async Task ExperimimentalTrainAndOptimize()
    {
        var experimentSettings = new RegressionExperimentSettings
        {
            MaxExperimentTimeInSeconds = 60,
            OptimizingMetric = RegressionMetric.RSquared
        };


        var experiment = mlContext.Auto().CreateRegressionExperiment(experimentSettings);

        var experimentResult = experiment.Execute(traingData, testData, labelColumnName: "CcsValue");

        model = experimentResult.BestRun.Model;
    }

    public override async Task SweepableTrainAndOptimize()
    {

        var sweepablePipeline = mlContext.Auto().Featurizer(traingData)
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "CcsValue", maximumNumberOfIterations: 100))
            .Append(mlContext.Auto().Regression(labelColumnName: "CcsValue"));

        var experimentSweepable = mlContext.Auto().CreateExperiment();

        experimentSweepable.SetPipeline(sweepablePipeline)
            .SetRegressionMetric(RegressionMetric.RSquared, labelColumn: "CcsValue")
            .SetTrainingTimeInSeconds(60)
            .SetDataset(traingData);

        // Log experiment trials
        mlContext.Log += (_, e) => {
            if (e.Source.Equals("AutoMLExperiment"))
            {
                Console.WriteLine(e.RawMessage);
            }
        };

        TrialResult experimentResults = await experimentSweepable.RunAsync();

        model = experimentResults.Model;
    }
    public async override Task TrainAsync()
    {

        var pipeline = mlContext.Transforms.CopyColumns("Label", "CcsValue")
            .Append(mlContext.Transforms.Concatenate("Features", GetFeatureColumnNames()))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "CcsValue", maximumNumberOfIterations: 100));

        model = await Task.Run(() => pipeline.Fit(traingData));
    }
}
