using CCSPredict.Data;
using CCSPredict.Descriptors;
using CCSPredict.Models.DataModels;
using com.epam.indigo;
using GraphMolWrap;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace CCSPredict.ML.PredictionModels;

public abstract class PredictionModel : IPredictionModel
{
    public IDataView data { get; set; }
    public TrainTestData dataSplit { get; private set; }
    public IDataView traingData { get; private set; }
    public IDataView testData { get; private set; }
    public MLContext mlContext { get; set; }

    private readonly IEnumerable<MoleculeData> moleculeData;

    public ITransformer model { get; set; }
    public CombinedFloatDescriptorCalculator descriptorCalculator { get; set; }
    public CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator { get; set; }

    protected PredictionModel(CombinedFloatDescriptorCalculator descriptorCalculator, CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator, IEnumerable<MoleculeData> moleculeData)
    {
        mlContext = new MLContext(seed: 0);

        this.moleculeData = moleculeData;

        this.descriptorCalculator = descriptorCalculator;
        this.bitVectorDescriptorCalculator = bitVectorDescriptorCalculator;
    }

    public static string[] GetFeatureColumnNames()
    {
        return new[]
        {
            "HallKierAlpha", 
            "Kappa1", 
            "Kappa2", 
            "Kappa3",
            "Chi0v", 
            "Chi1v", 
            "Chi2v", 
            "Chi3v", 
            "TPSA", 
            "LabuteASA",
            "MolecularWeight", 
            "NumHeavyAtoms", 
            "FractionCSP3",
            "MorganFingerprint", 
            "MACCSFingerprint",
            //"AtomPairFingerprint"
        };
    }

    public double CalculateConfidence(CcsPrediction prediction)
    {
        // Implement a method to calculate confidence based on the prediction
        return 0.95; // Placeholder value
    }

    
    public void SetData(IDataView data, double testFraction)
    {
        TrainTestData trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: testFraction);

        SetData(trainTestData.TestSet, trainTestData.TrainSet);
    }

    public void SetData(IDataView testData, IDataView trainingData)
    {
        this.testData = testData;
        this.traingData = trainingData;
    }

    public void LoadModel(string filePath)
    {
        model = mlContext.Model.Load(filePath, out var _);
    }

    public void SaveModel(string filePath)
    {
        mlContext.Model.Save(model, null, filePath);
    }

    public void SaveOnnxModel(string filePath = "./model.onnx")
    {
        using FileStream stream = File.Create(filePath);
        mlContext.Model.ConvertToOnnx(model, data, stream);
        Console.WriteLine($"Model saved to {filePath}");
    }

    public async Task TrainAsync(double testFraction)
    {
        data = mlContext.Data.LoadFromEnumerable(moleculeData);
        dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: testFraction);

        traingData = dataSplit.TrainSet;
        testData = dataSplit.TestSet;
    }

    public async Task<ModelMetrics> EvaluateAsync()
    {
        var predictions = model.Transform(testData);
        var metrics = mlContext.Regression.Evaluate(predictions);

        return new ModelMetrics
        {
            RSquared = metrics.RSquared,
            MeanAbsoluteError = metrics.MeanAbsoluteError,
            MeanSquaredError = metrics.MeanSquaredError,
            RootMeanSquaredError = metrics.RootMeanSquaredError
        };
    }

    public async Task<CcsPredictionResult> PredictAsync(Molecule molecule)
    {
        var descriptors = await CalculateDescriptorsAsync(molecule);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<MoleculeData, CcsPrediction>(model);
        var prediction = predictionEngine.Predict(descriptors);

        return new CcsPredictionResult
        {
            PredictedCcs = new CcsValue { Value = prediction.CcsValue, Unit = "Å²" },
            Confidence = CalculateConfidence(prediction)
        };
    }

    private async Task<MoleculeData> CalculateDescriptorsAsync(Molecule molecule)
    {
        var descriptorCalculator = new CombinedFloatDescriptorCalculator();
        var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(molecule);
        var bitVectorDescriptors = await bitVectorDescriptorCalculator.CalculateDescriptorsAsync(molecule); 

        return new MoleculeData
        {
            HallKierAlpha = (float)descriptors["HallKierAlpha"],
            Kappa1 = (float)descriptors["Kappa1"],
            Kappa2 = (float)descriptors["Kappa2"],
            Kappa3 = (float)descriptors["Kappa3"],
            Chi0v = (float)descriptors["Chi0v"],
            Chi1v = (float)descriptors["Chi1v"],
            Chi2v = (float)descriptors["Chi2v"],
            Chi3v = (float)descriptors["Chi3v"],
            TPSA = (float)descriptors["TPSA"],
            LabuteASA = (float)descriptors["LabuteASA"],
            MolecularWeight = (float)descriptors["ExactMolWt"],
            NumHeavyAtoms = (float)descriptors["NumHeavyAtoms"],
            FractionCSP3 = (float)descriptors["FractionCSP3"],
            MorganFingerprint = new VBuffer<float>(bitVectorDescriptors["MorganFingerprint"].Count, bitVectorDescriptors["MorganFingerprint"].ToArray()),
            MACCSFingerprint = new VBuffer<float>(bitVectorDescriptors["MACCSFingerprint"].Count, bitVectorDescriptors["MACCSFingerprint"].ToArray()),
            //AtomPairFingerprint = new VBuffer<float>(bitVectorDescriptors["AtomPairFingerprint"].Count, bitVectorDescriptors["AtomPairFingerprint"].ToArray()),
            //TopologicalTorsionFingerprint = new VBuffer<float>(bitVectorDescriptors["TopologicalTorsionFingerprint"].Count, bitVectorDescriptors["TopologicalTorsionFingerprint"].ToArray())
        };
    }
}
