using CCSPredict.Data;
using CCSPredict.Descriptors;
using CCSPredict.ML.PredictionModels;
using CCSPredict.Models.DataModels;
using com.epam.indigo;
using GraphMolWrap;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using static TorchSharp.torch.utils;

namespace CCSPredict.ML;

public class CcsPredictor
{
    private readonly FastTreePredictionModel model;
    private readonly SvmModel svmModel;
    private readonly RandomForestModel randomForestModel;
    private readonly NeuralNetworkModel neuralNetworkModel;

    private readonly CombinedFloatDescriptorCalculator descriptorCalculator;
    private readonly CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator;

    public CcsPredictor(ICcsDataProvider[] dataProviders)
    {
        descriptorCalculator = new CombinedFloatDescriptorCalculator();
        bitVectorDescriptorCalculator = new CombinedBitVectorDescriptorCalculator();
        
        List<MoleculeData> moleculeData = new List<MoleculeData>();

        foreach (ICcsDataProvider provider in dataProviders)
        {
            moleculeData.AddRange(provideData(provider).Result);
        }

        model = new FastTreePredictionModel(descriptorCalculator, bitVectorDescriptorCalculator, moleculeData);
        svmModel = new SvmModel(descriptorCalculator, bitVectorDescriptorCalculator, moleculeData);
        randomForestModel = new RandomForestModel(descriptorCalculator, bitVectorDescriptorCalculator, moleculeData);
        neuralNetworkModel = new NeuralNetworkModel(descriptorCalculator, bitVectorDescriptorCalculator, moleculeData);
    }

    private async Task<IEnumerable<MoleculeData>> provideData(ICcsDataProvider dataProvider)
    {
        var trainingData = await dataProvider.GetTrainingDataAsync();
        var testingData = await dataProvider.GetTestDataAsync();

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
                    ExtraInchiReturnValues extra = new ExtraInchiReturnValues();
                    RWMol mol = RDKFuncs.InchiToMol(molecule.InChI, extra);

                    molecule.Smiles = mol.MolToSmiles();

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

        return moleculeData;
    }

    public async Task<IEnumerable<MoleculeData>> PrepareDataAsync(IEnumerable<MoleculeWithCcs> molecules)
    {
        int index = 0;

        var moleculeDataTasks = molecules.Where(m => m.Adduct == "[M+H]+" || m.Adduct.Contains("H2O")).Select(async m =>
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
                //AtomPairFingerprint = new VBuffer<float>(bitVectorDescriptors["AtomPairFingerprint"].Count, bitVectorDescriptors["AtomPairFingerprint"].ToArray()),
                //TopologicalTorsionFingerprint = new VBuffer<float>(bitVectorDescriptors["TopologicalTorsionFingerprint"].Count, bitVectorDescriptors["TopologicalTorsionFingerprint"].ToArray()),


                CcsValue = (float)m.CcsValue
            };
        });

        return await Task.WhenAll(moleculeDataTasks);
    }


    public async Task TrainAndEvaluateAsync()
    {
        double testFraction = 0.2;

        Console.WriteLine("Training the model...");
        model.PrepareTrainingAndEvaluationData(testFraction);
        await model.SweepableTrainAndOptimize();

        Console.WriteLine("Training the SVM model...");
        svmModel.PrepareTrainingAndEvaluationData(testFraction);
        await svmModel.TrainAsync();
        //await svmModel.OptimizeParametersAsync();

        Console.WriteLine("Training the Random Forest model...");
        randomForestModel.PrepareTrainingAndEvaluationData(testFraction);
        await randomForestModel.TrainAsync();

        Console.WriteLine("Training the Neural Network model...");
        neuralNetworkModel.PrepareTrainingAndEvaluationData(testFraction);
        await neuralNetworkModel.TrainAsync();

        Console.WriteLine("Evaluating the model...");
        var metrics = await model.EvaluateAsync();

        Console.WriteLine("Evaluatin the svm model...");
        var svmMetrics = await svmModel.EvaluateAsync();

        Console.WriteLine("Evaluating the Random Forest model...");
        var randomForestMetrics = await randomForestModel.EvaluateAsync();

        Console.WriteLine("Evaluating the Neural Network model...");
        var neuralNetworkMetrics = await neuralNetworkModel.EvaluateAsync();

        Console.WriteLine("");
        Console.WriteLine($"Model Metrics:");
        Console.WriteLine($"R-squared: {metrics.RSquared}");
        Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
        Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

        Console.WriteLine("");
        Console.WriteLine($"SVM Model Metrics:");
        Console.WriteLine($"R-squared: {svmMetrics.RSquared}");
        Console.WriteLine($"Mean Absolute Error: {svmMetrics.MeanAbsoluteError}");
        Console.WriteLine($"Root Mean Squared Error: {svmMetrics.RootMeanSquaredError}");

        Console.WriteLine("");
        Console.WriteLine($"Random Forest Model Metrics:");
        Console.WriteLine($"R-squared: {randomForestMetrics.RSquared}");
        Console.WriteLine($"Mean Absolute Error: {randomForestMetrics.MeanAbsoluteError}");
        Console.WriteLine($"Root Mean Squared Error: {randomForestMetrics.RootMeanSquaredError}");

        Console.WriteLine("");
        Console.WriteLine($"Neural Network Model Metrics:");
        Console.WriteLine($"R-squared: {neuralNetworkMetrics.RSquared}");
        Console.WriteLine($"Mean Absolute Error: {neuralNetworkMetrics.MeanAbsoluteError}");
        Console.WriteLine($"Root Mean Squared Error: {neuralNetworkMetrics.RootMeanSquaredError}");

        Console.WriteLine("");
        svmModel.SaveModel("svm_ccs_prediction_model.zip");
        Console.WriteLine("Model saved to svm_ccs_prediction_model.zip");

        svmModel.SaveOnnxModel("./svm_model.onnx");  
        Console.WriteLine("Model saved to svm_ccs_prediction_model.onnx");

        model.SaveModel("ccs_prediction_model.zip");
        Console.WriteLine("Model saved to ccs_prediction_model.zip");
        model.SaveOnnxModel("./fast_model.onnx");

        randomForestModel.SaveModel("random_forest_ccs_prediction_model.zip");
        Console.WriteLine("Model saved to random_forest_ccs_prediction_model.zip");
        randomForestModel.SaveOnnxModel("./randomForest_model.onnx");

        neuralNetworkModel.SaveModel("neural_network_ccs_prediction_model.zip");
        Console.WriteLine("Model saved to neural_network_ccs_prediction_model.zip");
        neuralNetworkModel.SaveOnnxModel("./neuronal_model.onnx");
    }

    public async Task<CcsPredictionResult> PredictCcsAsync(string smiles, string inchi = null)
    {
        var molecule = new Molecule(smiles, inchi);
        return await model.PredictAsync(molecule);
    }
    public async Task<CcsPredictionResult> PredictCcsSVMAsync(string smiles, string inchi = null)
    {
        var molecule = new Molecule(smiles, inchi);
        return await svmModel.PredictAsync(molecule);
    }

    public async Task<CcsPredictionResult> PredictCcsRandomForestAsync(string smiles, string inchi = null)
    {
        var molecule = new Molecule(smiles, inchi);
        return await randomForestModel.PredictAsync(molecule);
    }

    public async Task<CcsPredictionResult> PredictCcsNeuralNetworkAsync(string smiles, string inchi = null)
    {
        var molecule = new Molecule(smiles, inchi);
        return await neuralNetworkModel.PredictAsync(molecule);
    }
}