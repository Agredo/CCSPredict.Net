

using CCSPredict.Data;
using CCSPredict.ML;

var dataProvider = new CsvCcsDataProvider("training_data.csv", "test_data.csv");
var predictor = new CcsPredictor(dataProvider);

await predictor.TrainAndEvaluateAsync();

Console.WriteLine("Enter SMILES to predict CCS (or 'exit' to quit):");
while (true)
{
    var smiles = Console.ReadLine();
    if (smiles.ToLower() == "exit") break;

    var prediction = await predictor.PredictCcsAsync(smiles);
    Console.WriteLine($"Predicted CCS: {prediction.PredictedCcs} {prediction.PredictedCcs}");
    Console.WriteLine($"Confidence: {prediction.Confidence}");
}
