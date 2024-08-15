using CCSPredict.Data;
using CCSPredict.Models.DataModels;

namespace CCSPredict.ML;

public class CcsPredictor
{
    private readonly CcsPredictionModel model;

    public CcsPredictor(ICcsDataProvider dataProvider)
    {
        model = new CcsPredictionModel(dataProvider);
    }

    public async Task TrainAndEvaluateAsync()
    {
        Console.WriteLine("Training the model...");
        await model.TrainAsync();

        Console.WriteLine("Evaluating the model...");
        var metrics = await model.EvaluateAsync();

        Console.WriteLine($"Model Metrics:");
        Console.WriteLine($"R-squared: {metrics.RSquared}");
        Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
        Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

        model.SaveModel("ccs_prediction_model.zip");
        Console.WriteLine("Model saved to ccs_prediction_model.zip");
        model.SaveOnnxModel();
        


    }

    public async Task<CcsPredictionResult> PredictCcsAsync(string smiles, string inchi = null)
    {
        var molecule = new Molecule(smiles, inchi);
        return await model.PredictAsync(molecule);
    }
}