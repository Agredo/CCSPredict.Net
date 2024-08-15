using CCSPredict.Data;
using CCSPredict.Models.DataModels;

namespace CCSPredict.ML;

public class CcsPredictor
{
    private readonly CcsPredictionModel model;
    private readonly SvmModel svmModel;
    private readonly RandomForestModel randomForestModel;

    public CcsPredictor(ICcsDataProvider dataProvider)
    {
        model = new CcsPredictionModel(dataProvider);
        svmModel = new SvmModel(dataProvider);
        randomForestModel = new RandomForestModel(dataProvider);
    }

    public async Task TrainAndEvaluateAsync()
    {
        Console.WriteLine("Training the model...");
        await model.TrainAsync();

        Console.WriteLine("Training the SVM model...");
        await svmModel.TrainAsync();

        Console.WriteLine("Training the Random Forest model...");
        await randomForestModel.TrainAsync();

        Console.WriteLine("Evaluating the model...");
        var metrics = await model.EvaluateAsync();

        Console.WriteLine("Evaluatin the svm model...");
        var svmMetrics = await svmModel.EvaluateAsync();

        Console.WriteLine("Evaluating the Random Forest model...");
        var randomForestMetrics = await randomForestModel.EvaluateAsync();

        Console.WriteLine($"Model Metrics:");
        Console.WriteLine($"R-squared: {metrics.RSquared}");
        Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
        Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

        Console.WriteLine($"SVM Model Metrics:");
        Console.WriteLine($"R-squared: {svmMetrics.RSquared}");
        Console.WriteLine($"Mean Absolute Error: {svmMetrics.MeanAbsoluteError}");
        Console.WriteLine($"Root Mean Squared Error: {svmMetrics.RootMeanSquaredError}");

        Console.WriteLine($"Random Forest Model Metrics:");
        Console.WriteLine($"R-squared: {randomForestMetrics.RSquared}");
        Console.WriteLine($"Mean Absolute Error: {randomForestMetrics.MeanAbsoluteError}");
        Console.WriteLine($"Root Mean Squared Error: {randomForestMetrics.RootMeanSquaredError}");

        svmModel.SaveModel("svm_ccs_prediction_model.zip");
        Console.WriteLine("Model saved to svm_ccs_prediction_model.zip");

        svmModel.SaveOnnxModel();  
        Console.WriteLine("Model saved to svm_ccs_prediction_model.onnx");

        model.SaveModel("ccs_prediction_model.zip");
        Console.WriteLine("Model saved to ccs_prediction_model.zip");
        model.SaveOnnxModel();

        randomForestModel.SaveModel("random_forest_ccs_prediction_model.zip");
        Console.WriteLine("Model saved to random_forest_ccs_prediction_model.zip");
        randomForestModel.SaveOnnxModel();




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
}