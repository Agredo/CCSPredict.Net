using CCSPredict.Data;
using CCSPredict.Descriptors;
using CCSPredict.Models.DataModels;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace CCSPredict.ML;

public class ModelOptimizer
{
    private readonly MLContext mlContext;
    private readonly ICcsDataProvider dataProvider;
    public CombinedFloatDescriptorCalculator descriptorCalculator { get; set; }
    public CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator { get; set; }

    public ModelOptimizer(ICcsDataProvider dataProvider)
    {
        mlContext = new MLContext(seed: 0);
        this.dataProvider = dataProvider;
        this.descriptorCalculator = new CombinedFloatDescriptorCalculator();
        this.bitVectorDescriptorCalculator = new CombinedBitVectorDescriptorCalculator();
    }

    public async Task<OptimizationResult> OptimizeModelAsync(string modelType, TimeSpan trainingTime)
    {
        var trainingData = await dataProvider.GetTrainingDataAsync();
        var testData = await dataProvider.GetTestDataAsync();

        var moleculeData = await PrepareDataAsync(trainingData.Concat(testData));
        var data = mlContext.Data.LoadFromEnumerable(moleculeData);

        var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        var experimentSettings = new RegressionExperimentSettings
        {
            MaxExperimentTimeInSeconds = (uint)trainingTime.TotalSeconds,
            OptimizingMetric = RegressionMetric.RSquared
        };

        ExperimentResult<RegressionMetrics> experimentResult = null;

        switch (modelType.ToLower())
        {
            case "fastforest":
                experimentSettings.Trainers.Clear();
                experimentSettings.Trainers.Add(RegressionTrainer.FastForest);
                experimentSettings.Trainers.Add(RegressionTrainer.FastTree);
                experimentSettings.Trainers.Add(RegressionTrainer.FastTreeTweedie);
                break;

            case "svm":
                //experimentSettings.Trainers.Clear();
                //var sdcaOption = new SdcaRegressionTrainer.Options
                //{
                //    LabelColumnName = "CcsValue",
                //    FeatureColumnName = "Features",
                //    MaximumNumberOfIterations = 100,
                //    L2Regularization = 0.1f,
                //    L1Regularization = 0.1f
                //};
                //experimentSettings.Trainers.Add(new SdcaRegressionTrainer(null, sdcaOption));
                break;

            case "neuralnetwork":
                experimentSettings.Trainers.Clear();
                experimentSettings.Trainers.Add(RegressionTrainer.LbfgsPoissonRegression);
                break;

            default:
                throw new ArgumentException("Unsupported model type", nameof(modelType));
        }

        experimentResult = mlContext.Auto()
            .CreateRegressionExperiment(experimentSettings).Execute(splitData.TrainSet);

        var bestRun = experimentResult.BestRun;
        var bestModel = bestRun.Model;
        var bestPipeline = bestRun.Estimator;
        var metrics = bestRun.ValidationMetrics;

        return new OptimizationResult
        {
            BestPipeline = bestPipeline,
            BestModel = bestModel,
            Metrics = new ModelMetrics
            {
                RSquared = metrics.RSquared,
                MeanAbsoluteError = metrics.MeanAbsoluteError,
                MeanSquaredError = metrics.MeanSquaredError,
                RootMeanSquaredError = metrics.RootMeanSquaredError
            }
        };
    }

    private async Task<IEnumerable<MoleculeData>> PrepareDataAsync(IEnumerable<MoleculeWithCcs> molecules)
    {
        int index = 0;
        var moleculeDataTasks = molecules.Where(m => m.Adduct == "[M+H]+").Select(async m =>
        {
            var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(new Molecule(m.Smiles, m.InChI));
            var bitVectorDescriptors = await bitVectorDescriptorCalculator.CalculateDescriptorsAsync(new Molecule(m.Smiles, m.InChI));
            Console.WriteLine($"Nr. {index++} - Adduct: {m.Adduct} CCS: {m.CcsValue} m/z {m.MZ}");
            return new MoleculeData
            {
                HallKierAlpha = descriptors["HallKierAlpha"],
                Kappa1 = descriptors["Kappa1"],
                Kappa2 = descriptors["Kappa2"],
                Kappa3 = descriptors["Kappa3"],
                Chi0v = descriptors["Chi0v"],
                Chi1v = descriptors["Chi1v"],
                Chi2v = descriptors["Chi2v"],
                Chi3v = descriptors["Chi3v"],
                TPSA = descriptors["TPSA"],
                LabuteASA = descriptors["LabuteASA"],
                MolecularWeight = descriptors["ExactMolWt"],
                NumHeavyAtoms = descriptors["NumHeavyAtoms"],
                FractionCSP3 = descriptors["FractionCSP3"],
                MorganFingerprint = new VBuffer<float>(bitVectorDescriptors["MorganFingerprint"].Count, bitVectorDescriptors["MorganFingerprint"].ToArray()),
                MACCSFingerprint = new VBuffer<float>(bitVectorDescriptors["MACCSFingerprint"].Count, bitVectorDescriptors["MACCSFingerprint"].ToArray()),
                CcsValue = (float)m.CcsValue
            };
        });

        return await Task.WhenAll(moleculeDataTasks);
    }
}

