using CCSPredict.Data;
using CCSPredict.Descriptors;
using CCSPredict.Models.DataModels;
using com.epam.indigo;
using GraphMolWrap;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace CCSPredict.ML.PredictionModels;

public abstract class PredictionModel : IPredictionModel
{
    public IDataView data { get; set; }
    public TrainTestData dataSplit { get; private set; }
    public IDataView traingData { get; private set; }
    public IDataView testData { get; private set; }
    public MLContext mlContext { get; set; }
    public ITransformer model { get; set; }
    public ICcsDataProvider dataProvider { get; set; }
    public CombinedFloatDescriptorCalculator descriptorCalculator { get; set; }
    public CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator { get; set; }

    protected PredictionModel(ICcsDataProvider dataProvider)
    {
        mlContext = new MLContext(seed: 0);
        this.dataProvider = dataProvider;
        descriptorCalculator = new CombinedFloatDescriptorCalculator();
        bitVectorDescriptorCalculator = new CombinedBitVectorDescriptorCalculator();
    }

    public static string[] GetFeatureColumnNames()
    {
        return new[]
        {
            "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
            "Chi0v", "Chi1v", "Chi2v", "Chi3v", "TPSA", "LabuteASA",
            "MolecularWeight", "NumHeavyAtoms", "FractionCSP3",
            "MorganFingerprint",
        };
    }

    public double CalculateConfidence(CcsPrediction prediction)
    {
        // Implement a method to calculate confidence based on the prediction
        return 0.95; // Placeholder value
    }

    public void LoadModel(string filePath)
    {
        model = mlContext.Model.Load(filePath, out var _);
    }

    public async Task<IEnumerable<MoleculeData>> PrepareDataAsync(IEnumerable<MoleculeWithCcs> molecules)
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
                //AtomPairFingerprint = descriptors["AtomPairFingerprint"],
                //TopologicalTorsionFingerprint = descriptors["TopologicalTorsionFingerprint"],
                CcsValue = (float)m.CcsValue
            };
        });

        return await Task.WhenAll(moleculeDataTasks);
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

    public async Task TrainAsync()
    {
        var trainingData = await dataProvider.GetTrainingDataAsync();
        var testingData = await dataProvider.GetTestDataAsync();


        //merge data and check for equal smiles and inchi
        //var mergedData = trainingData.Concat(testingData).ToList();

        var mergedData = new List<MoleculeWithCcs>();
        mergedData.AddRange(trainingData);
        mergedData.AddRange(testingData);


        var dataWithSmiles = new List<MoleculeWithCcs>();

        Indigo indigo = new Indigo();
        foreach (var molecule in mergedData)
        {
            if (!string.IsNullOrEmpty(molecule.Smiles))
            {
                dataWithSmiles.Add(molecule);
            }
            else if (string.IsNullOrEmpty(molecule.Smiles) && !string.IsNullOrEmpty(molecule.InChI))
            {
                try
                {
                    //Console.WriteLine($"Converting InChI to SMILES for molecule {molecule.InChI}");

                    ExtraInchiReturnValues extra = new ExtraInchiReturnValues();
                    RWMol mol = RDKFuncs.InchiToMol(molecule.InChI, extra);

                    molecule.Smiles = mol.MolToSmiles();
                    //molecule.Smiles = indigo.loadMolecule(molecule.InChI).canonicalSmiles();

                    dataWithSmiles.Add(molecule);

                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to convert InChI to SMILES for molecule {molecule.InChI}: {ex.Message}");
                }
            }
        }

        IEnumerable<MoleculeData> moleculeData;

        if (dataWithSmiles.Count > 0)
        {
            moleculeData = await PrepareDataAsync(dataWithSmiles);
        }

        else
        {
            moleculeData = await PrepareDataAsync(mergedData);

        }

        data = mlContext.Data.LoadFromEnumerable(moleculeData);

        dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.01);

        traingData = dataSplit.TrainSet;
        testData = dataSplit.TestSet;
    }

    public async Task<ModelMetrics> EvaluateAsync()
    {
        //var testData = await dataProvider.GetTestDataAsync();
        //var dataWithSmiles = new List<MoleculeWithCcs>();

        //Indigo indigo = new Indigo();
        //foreach (var molecule in testData)
        //{
        //    if(string.IsNullOrEmpty(molecule.Smiles) && !string.IsNullOrEmpty(molecule.InChI))
        //    {
        //        try
        //        {
        //            //Console.WriteLine($"Converting InChI to SMILES for molecule {molecule.InChI}");

        //            ExtraInchiReturnValues extra = new ExtraInchiReturnValues();
        //            RWMol mol = RDKFuncs.InchiToMol(molecule.InChI, extra);

        //            molecule.Smiles = mol.MolToSmiles();
        //            //molecule.Smiles = indigo.loadMolecule(molecule.InChI).canonicalSmiles();

        //            dataWithSmiles.Add(molecule);
        //        }
        //        catch (Exception ex)
        //        {
        //            Console.WriteLine($"Failed to convert InChI to SMILES for molecule {molecule.InChI}: {ex.Message}");
        //        }
        //    }
        //}

        //if (dataWithSmiles.Count() > 0)
        //{
        //    var moleculeData = await PrepareDataAsync(dataWithSmiles);

        //    data = mlContext.Data.LoadFromEnumerable(moleculeData);
        //}
        //else
        //{
        //    var moleculeData = await PrepareDataAsync(testData);

        //    data = mlContext.Data.LoadFromEnumerable(moleculeData);
        //}

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
            //AtomPairFingerprint = (float)descriptors["AtomPairFingerprint"],
            //TopologicalTorsionFingerprint = (float)descriptors["TopologicalTorsionFingerprint"]
        };
    }
}
