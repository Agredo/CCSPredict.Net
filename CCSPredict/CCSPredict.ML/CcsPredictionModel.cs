using CCSPredict.Data;
using CCSPredict.Descriptors;
using CCSPredict.Models.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace CCSPredict.ML;

public class CcsPredictionModel
{
    private readonly MLContext mlContext;
    private ITransformer model;
    private readonly ICcsDataProvider dataProvider;
    private readonly CombinedDescriptorCalculator descriptorCalculator;

    public CcsPredictionModel(ICcsDataProvider dataProvider)
    {
        mlContext = new MLContext(seed: 0);
        this.dataProvider = dataProvider;
        descriptorCalculator = new CombinedDescriptorCalculator();
    }

    public async Task TrainAsync()
    {
        var trainingData = await dataProvider.GetTrainingDataAsync();
        var moleculeData = await PrepareDataAsync(trainingData);

        var data = mlContext.Data.LoadFromEnumerable(moleculeData);

        var featureColumnNames = moleculeData.First().Descriptors.Keys.ToArray();
        var pipeline = mlContext.Transforms.Concatenate("Features", featureColumnNames)
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "CcsValue"));

        model = pipeline.Fit(data);
    }

    public async Task<CcsPredictionResult> PredictAsync(Molecule molecule)
    {
        var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(molecule);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<MoleculeData, CcsPrediction>(model);
        var moleculeData = new MoleculeData { Descriptors = descriptors };
        var prediction = predictionEngine.Predict(moleculeData);

        return new CcsPredictionResult
        {
            PredictedCcs = new CcsValue { Value = prediction.CcsValue, Unit = "Å²" },
            Confidence = CalculateConfidence(prediction)
        };
    }

    public async Task<ModelMetrics> EvaluateAsync()
    {
        var testData = await dataProvider.GetTestDataAsync();
        var moleculeData = await PrepareDataAsync(testData);

        var data = mlContext.Data.LoadFromEnumerable(moleculeData);
        var predictions = model.Transform(data);
        var metrics = mlContext.Regression.Evaluate(predictions);

        return new ModelMetrics
        {
            RSquared = metrics.RSquared,
            MeanAbsoluteError = metrics.MeanAbsoluteError,
            MeanSquaredError = metrics.MeanSquaredError,
            RootMeanSquaredError = metrics.RootMeanSquaredError
        };
    }

    private async Task<IEnumerable<MoleculeData>> PrepareDataAsync(IEnumerable<MoleculeWithCcs> molecules)
    {
        var moleculeDataTasks = molecules.Select(async m =>
        {
            var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(new Molecule(m.Smiles, m.InChI));
            return new MoleculeData
            {
                Descriptors = descriptors,
                CcsValue = (float)m.CcsValue
            };
        });

        return await Task.WhenAll(moleculeDataTasks);
    }

    private double CalculateConfidence(CcsPrediction prediction)
    {
        // Implement a method to calculate confidence based on the prediction
        return 0.95; // Placeholder value
    }

    public void SaveModel(string filePath)
    {
        mlContext.Model.Save(model, null, filePath);
    }

    public void LoadModel(string filePath)
    {
        model = mlContext.Model.Load(filePath, out var _);
    }
}

public class CcsValue
{
    public float Value { get; set; }
    public string Unit { get; set; }
}

public class ModelMetrics
{
    public double MeanAbsoluteError { get; internal set; }
    public double RSquared { get; internal set; }
    public double MeanSquaredError { get; internal set; }
    public double RootMeanSquaredError { get; internal set; }
}

public class CcsPredictionResult
{
    public CcsValue PredictedCcs { get; set; }
    public double Confidence { get; set; }
}

public class MoleculeData
{
    [VectorType(1)]
    public Dictionary<string, double> Descriptors { get; set; }

    public float CcsValue { get; set; }
}

public class CcsPrediction
{
    [ColumnName("Score")]
    public float CcsValue;
}