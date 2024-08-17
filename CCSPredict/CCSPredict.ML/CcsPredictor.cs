using CCSPredict.Data;
using CCSPredict.ML.PredictionModels;
using CCSPredict.Models.DataModels;
using Microsoft.ML;

namespace CCSPredict.ML;

public class CcsPredictor
{
    private readonly FastTreePredictionModel model;
    private readonly SvmModel svmModel;
    private readonly RandomForestModel randomForestModel;
    private readonly NeuralNetworkModel neuralNetworkModel;

    public CcsPredictor(ICcsDataProvider dataProvider)
    {
        model = new FastTreePredictionModel(dataProvider);
        svmModel = new SvmModel(dataProvider);
        randomForestModel = new RandomForestModel(dataProvider);
        neuralNetworkModel = new NeuralNetworkModel(dataProvider);
    }


    public async Task TrainAndEvaluateAsync()
    {
        Console.WriteLine("Training the model...");
        await model.TrainAsync();

        Console.WriteLine("Training the SVM model...");
        await svmModel.TrainAsync();
        //await svmModel.OptimizeParametersAsync();

        Console.WriteLine("Training the Random Forest model...");
        await randomForestModel.TrainAsync();

        Console.WriteLine("Training the Neural Network model...");
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