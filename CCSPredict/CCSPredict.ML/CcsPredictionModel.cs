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

    public IDataView data { get; private set; }

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

        data = mlContext.Data.LoadFromEnumerable(moleculeData);

        var featureColumnNames = new[]
        {
            "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
            "Chi0v", "Chi1v", "Chi2v", "Chi3v", "TPSA", "LabuteASA", "NumRadicalElectrons", "MaxPartialCharge", "MinPartialCharge", "MeanPartialCharge",
            "ExactMolWt", "NumRotatableBonds","NumHeavyAtoms","FractionCSP3", "MorganFingerprint", "AtomPairFingerprint", "TopologicalTorsionFingerprint"
        };

        var pipeline = mlContext.Transforms.CopyColumns("Label", "CcsValue")
            .Append(mlContext.Transforms.Concatenate("Features", featureColumnNames))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "CcsValue"));

        model = pipeline.Fit(data);
    }

    public async Task<CcsPredictionResult> PredictAsync(Molecule molecule)
    {
        var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(molecule);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<MoleculeData, CcsPrediction>(model);
        var moleculeData = new MoleculeData
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
            //MaxPartialCharge = descriptors["MaxPartialCharge"],
            //MinPartialCharge = descriptors["MinPartialCharge"],
            //MeanPartialCharge = descriptors["MeanPartialCharge"],
            //NumRadicalElectrons = descriptors["NumRadicalElectrons"],
            MolecularWeight = descriptors["ExactMolWt"],
            NumHeavyAtoms = descriptors["NumHeavyAtoms"],
            FractionCSP3 = descriptors["FractionCSP3"],
            MorganFingerprint = descriptors["MorganFingerprint"],
            AtomPairFingerprint = descriptors["AtomPairFingerprint"],
            TopologicalTorsionFingerprint = descriptors["TopologicalTorsionFingerprint"],
        };
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
        int index = 0;
        var moleculeDataTasks = molecules.Where(m => m.Adduct == "[M+H]+").Select(async m =>
        {
            var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(new Molecule(m.Smiles, m.InChI));
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
                //MaxPartialCharge = descriptors["MaxPartialCharge"],
                //MinPartialCharge = descriptors["MinPartialCharge"],
                //MeanPartialCharge = descriptors["MeanPartialCharge"],
                //NumRadicalElectrons = descriptors["NumRadicalElectrons"],
                MolecularWeight = descriptors["ExactMolWt"],
                NumHeavyAtoms = descriptors["NumHeavyAtoms"],
                FractionCSP3 = descriptors["FractionCSP3"],
                MorganFingerprint = descriptors["MorganFingerprint"],
                AtomPairFingerprint = descriptors["AtomPairFingerprint"],
                TopologicalTorsionFingerprint = descriptors["TopologicalTorsionFingerprint"],

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

    public void SaveOnnxModel(string filePath = "./model.onnx")
    {
        using FileStream stream = File.Create(filePath);
        mlContext.Model.ConvertToOnnx(model, data, stream);
        Console.WriteLine($"Model saved to {filePath}");
    }


    public void LoadModel(string filePath)
    {
        model = mlContext.Model.Load(filePath, out var _);
    }
}

public class SvmModel
{
    private readonly MLContext mlContext;
    private ITransformer model;
    private readonly ICcsDataProvider dataProvider;
    private readonly CombinedDescriptorCalculator descriptorCalculator;

    public IDataView data { get; private set; }

    public SvmModel(ICcsDataProvider dataProvider)
    {
        mlContext = new MLContext(seed: 0);
        this.dataProvider = dataProvider;
        descriptorCalculator = new CombinedDescriptorCalculator();
    }

    public async Task TrainAsync()
    {
        var trainingData = await dataProvider.GetTrainingDataAsync();
        var moleculeData = await PrepareDataAsync(trainingData);

        data = mlContext.Data.LoadFromEnumerable(moleculeData);

        var pipeline = mlContext.Transforms.CopyColumns("Label", "CcsValue")
            .Append(mlContext.Transforms.Concatenate("Features", GetFeatureColumnNames()))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "CcsValue", maximumNumberOfIterations: 100));

        model = await Task.Run(() => pipeline.Fit(data));
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
        int index = 0;
        var moleculeDataTasks = molecules.Where(m => m.Adduct == "[M+H]+").Select(async m =>
        {
            var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(new Molecule(m.Smiles, m.InChI));
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
                //MaxPartialCharge = descriptors["MaxPartialCharge"],
                //MinPartialCharge = descriptors["MinPartialCharge"],
                //MeanPartialCharge = descriptors["MeanPartialCharge"],
                //NumRadicalElectrons = descriptors["NumRadicalElectrons"],
                MolecularWeight = descriptors["ExactMolWt"],
                NumHeavyAtoms = descriptors["NumHeavyAtoms"],
                FractionCSP3 = descriptors["FractionCSP3"],
                MorganFingerprint = descriptors["MorganFingerprint"],
                AtomPairFingerprint = descriptors["AtomPairFingerprint"],
                TopologicalTorsionFingerprint = descriptors["TopologicalTorsionFingerprint"],

                CcsValue = (float)m.CcsValue
            };
        });

        return await Task.WhenAll(moleculeDataTasks);
    }

    public void SaveModel(string filePath)
    {
        mlContext.Model.Save(model, null, filePath);
    }

    public void SaveOnnxModel(string filePath = "./svm_ccs_prediction_model.onnx")
    {
        using FileStream stream = File.Create(filePath);
        mlContext.Model.ConvertToOnnx(model, data, stream);
        Console.WriteLine($"Model saved to {filePath}");
    }

    public void LoadModel(string filePath)
    {
        model = mlContext.Model.Load(filePath, out var _);
    }

    private async Task<MoleculeData> CalculateDescriptorsAsync(Molecule molecule)
    {
        var descriptorCalculator = new CombinedDescriptorCalculator();
        var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(molecule);

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
            MorganFingerprint = (float)descriptors["MorganFingerprint"],
            AtomPairFingerprint = (float)descriptors["AtomPairFingerprint"],
            TopologicalTorsionFingerprint = (float)descriptors["TopologicalTorsionFingerprint"]
        };
    }

    private double CalculateConfidence(CcsPrediction prediction)
    {
        // Implement a method to calculate confidence based on the prediction
        // This is a placeholder and should be replaced with a proper confidence calculation
        return 0.95;
    }

    private static string[] GetFeatureColumnNames()
    {
        return new[]
        {
                "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
                "Chi0v", "Chi1v", "Chi2v", "Chi3v", "TPSA", "LabuteASA",
                "MolecularWeight", "NumHeavyAtoms", "FractionCSP3",
                "MorganFingerprint", "AtomPairFingerprint", "TopologicalTorsionFingerprint"
            };
    }
}

public class RandomForestModel
{
    private readonly MLContext mlContext;
    private ITransformer model;
    private readonly ICcsDataProvider dataProvider;
    private readonly CombinedDescriptorCalculator descriptorCalculator;

    public IDataView data { get; private set; }

    public RandomForestModel(ICcsDataProvider dataProvider)
    {
        mlContext = new MLContext(seed: 0);
        this.dataProvider = dataProvider;
        descriptorCalculator = new CombinedDescriptorCalculator();
    }

    public async Task TrainAsync()
    {
        var trainingData = await dataProvider.GetTrainingDataAsync();
        var moleculeData = await PrepareDataAsync(trainingData);

        data = mlContext.Data.LoadFromEnumerable(moleculeData);

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
        int index = 0;
        var moleculeDataTasks = molecules.Where(m => m.Adduct == "[M+H]+").Select(async m =>
        {
            var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(new Molecule(m.Smiles, m.InChI));
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
                MorganFingerprint = descriptors["MorganFingerprint"],
                AtomPairFingerprint = descriptors["AtomPairFingerprint"],
                TopologicalTorsionFingerprint = descriptors["TopologicalTorsionFingerprint"],
                CcsValue = (float)m.CcsValue
            };
        });

        return await Task.WhenAll(moleculeDataTasks);
    }

    public void SaveModel(string filePath)
    {
        mlContext.Model.Save(model, null, filePath);
    }

    public void SaveOnnxModel(string filePath = "./random_forest_ccs_prediction_model.onnx")
    {
        using FileStream stream = File.Create(filePath);
        mlContext.Model.ConvertToOnnx(model, data, stream);
        Console.WriteLine($"Model saved to {filePath}");
    }

    public void LoadModel(string filePath)
    {
        model = mlContext.Model.Load(filePath, out var _);
    }

    private async Task<MoleculeData> CalculateDescriptorsAsync(Molecule molecule)
    {
        var descriptorCalculator = new CombinedDescriptorCalculator();
        var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(molecule);

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
            MorganFingerprint = (float)descriptors["MorganFingerprint"],
            AtomPairFingerprint = (float)descriptors["AtomPairFingerprint"],
            TopologicalTorsionFingerprint = (float)descriptors["TopologicalTorsionFingerprint"]
        };
    }

    private double CalculateConfidence(CcsPrediction prediction)
    {
        // Implement a method to calculate confidence based on the prediction
        // This is a placeholder and should be replaced with a proper confidence calculation
        return 0.95;
    }

    private static string[] GetFeatureColumnNames()
    {
        return new[]
        {
            "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
            "Chi0v", "Chi1v", "Chi2v", "Chi3v", "TPSA", "LabuteASA",
            "MolecularWeight", "NumHeavyAtoms", "FractionCSP3",
            "MorganFingerprint", "AtomPairFingerprint", "TopologicalTorsionFingerprint"
        };
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
    //[ColumnName("Descriptor1")]
    //public double Descriptor1 { get; set; }

    //[VectorType(2)]
    //public Dictionary<string, double> Descriptors { get; set; }

    [ColumnName("HallKierAlpha")]
    public float HallKierAlpha { get; set; }

    [ColumnName("Kappa1")]
    public float Kappa1 { get; set; }

    [ColumnName("Kappa2")]
    public float Kappa2 { get; set; }

    [ColumnName("Kappa3")]
    public float Kappa3 { get; set; }

    [ColumnName("Chi0v")]
    public float Chi0v { get; set; }

    [ColumnName("Chi1v")]
    public float Chi1v { get; set; }

    [ColumnName("Chi2v")]
    public float Chi2v { get; set; }

    [ColumnName("Chi3v")]
    public float Chi3v { get; set; }

    [ColumnName("MolecularWeight")]
    public float MolecularWeight { get; set; }

    [ColumnName("TPSA")]
    public float TPSA { get; set; }

    [ColumnName("LogP")]
    public float LogP { get; set; }

    [ColumnName("LabuteASA")]
    public float LabuteASA { get; set; }

    [ColumnName("NumRadicalElectrons")]
    public float NumRadicalElectrons { get; set; }

    [ColumnName("MaxPartialCharge")]
    public float MaxPartialCharge { get; set; }

    [ColumnName("MinPartialCharge")]
    public float MinPartialCharge { get; set; }

    [ColumnName("MeanPartialCharge")]
    public float MeanPartialCharge { get; set; }

    [ColumnName("FractionCSP3")]
    public float FractionCSP3 { get; set; }

    [ColumnName("ExactMolWt")]
    public float ExactMolWt { get; set; }

    [ColumnName("NumRotatableBonds")]
    public float NumRotatableBonds { get; set; }

    [ColumnName("NumHeavyAtoms")]
    public float NumHeavyAtoms { get; set; }

    [ColumnName("MorganFingerprint")]
    public float MorganFingerprint { get; set; }

    [ColumnName("AtomPairFingerprint")]
    public float AtomPairFingerprint { get; set; }

    [ColumnName("TopologicalTorsionFingerprint")]
    public float TopologicalTorsionFingerprint { get; set; }

    [ColumnName("CcsValue")]
    public float CcsValue { get; set; }


}

public class CcsPrediction
{
    [ColumnName("Score")]
    public float CcsValue;
}