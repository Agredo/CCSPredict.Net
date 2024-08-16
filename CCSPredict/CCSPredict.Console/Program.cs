

using CCSPredict.Data;
using CCSPredict.ML;

var dataProvider = new CsvCcsDataProvider("C:\\Users\\chris\\Documents\\training.csv", "C:\\Users\\chris\\Documents\\test.csv");
var predictor = new CcsPredictor(dataProvider);

await predictor.TrainAndEvaluateAsync();

Console.WriteLine("Enter SMILES to predict CCS (or 'exit' to quit):");
while (true)
{
    var smiles = Console.ReadLine();
    if (smiles.ToLower() == "exit") break;

    var prediction = await predictor.PredictCcsAsync(smiles);
    var svmPrediction = await predictor.PredictCcsSVMAsync(smiles);
    var randomForestPrediction = await predictor.PredictCcsRandomForestAsync(smiles);
    var neuralNetworkPrediction = await predictor.PredictCcsNeuralNetworkAsync(smiles);

    Console.WriteLine($"Predicted CCS (FastTreeTweedie): {prediction.PredictedCcs.Value} {prediction.PredictedCcs.Unit}");
    Console.WriteLine($"Predicted CCS (SVM): {svmPrediction.PredictedCcs.Value} {svmPrediction.PredictedCcs.Unit}");
    Console.WriteLine($"Predicted CCS (Random Forest): {randomForestPrediction.PredictedCcs.Value} {randomForestPrediction.PredictedCcs.Unit}");
    Console.WriteLine($"Predicted CCS (Neural Network): {neuralNetworkPrediction.PredictedCcs.Value} {neuralNetworkPrediction.PredictedCcs.Unit}");
    //Console.WriteLine($"Confidence: {prediction.Confidence}");

}
