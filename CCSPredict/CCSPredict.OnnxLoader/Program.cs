using CCSPredict.ML;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

class Program
{
    private static DescriptorsCalculator DescriptorsCalculator { get; set; } = new DescriptorsCalculator();
    static void Main(string[] args)
    {
        string basePath = "C:\\Projects\\CCSPredict.Net\\CCSPredict\\CCSPredict.Console\\bin\\x64\\Debug\\net8.0\\";

        var mlContext = new MLContext();

        // Laden Sie die ONNX-Modelle
        var fastTreeModel = LoadOnnxModel(mlContext, basePath + "fast_model.onnx");
        var svmModel = LoadOnnxModel(mlContext, basePath + "svm_model.onnx");
        var randomForestModel = LoadOnnxModel(mlContext, basePath + "randomForest_model.onnx");
        var neuralNetworkModel = LoadOnnxModel(mlContext, basePath + "neuronal_model.onnx");

        Console.WriteLine("Enter SMILES to predict CCS (or 'exit' to quit):");

        while (true)
        {
            var input = Console.ReadLine();
            if (input.ToLower() == "exit") break;

            var moleculeData = CalculateDescriptors(input);

            var fastTreePrediction = PredictCcs(mlContext, fastTreeModel, moleculeData);
            var svmPrediction = PredictCcs(mlContext, svmModel, moleculeData);
            var randomForestPrediction = PredictCcs(mlContext, randomForestModel, moleculeData);
            var neuralNetworkPrediction = PredictCcs(mlContext, neuralNetworkModel, moleculeData);

            Console.WriteLine($"Predicted CCS (FastTree): {fastTreePrediction} Å²");
            Console.WriteLine($"Predicted CCS (SVM): {svmPrediction} Å²");
            Console.WriteLine($"Predicted CCS (Random Forest): {randomForestPrediction} Å²");
            Console.WriteLine($"Predicted CCS (Neural Network): {neuralNetworkPrediction} Å²");
        }
    }

    static ITransformer LoadOnnxModel(MLContext mlContext, string modelPath)
    {
        var pipeline = mlContext.Transforms.ApplyOnnxModel(modelPath);
        return pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<MoleculeData>()));
    }

    static float PredictCcs(MLContext mlContext, ITransformer model, MoleculeData moleculeData)
    {
        var predictionEngine = mlContext.Model.CreatePredictionEngine<MoleculeData, CcsPrediction>(model);
        var prediction = predictionEngine.Predict(moleculeData);
        return prediction.CcsValue;
    }

    static MoleculeData CalculateDescriptors(string smiles)
    {
        // Implementieren Sie hier Ihre Logik zur Berechnung der Deskriptoren
        // Dies sollte eine MoleculeData-Instanz zurückgeben
        // Beispiel:
        return DescriptorsCalculator.CalculateDescriptorsAsync(smiles, "").Result;
    }
}

class OnnxModel
{
    private readonly InferenceSession _session;

    public OnnxModel(string modelPath)
    {
        _session = new InferenceSession(modelPath);
    }

    public float Predict(float[] input)
    {
        var inputTensor = new DenseTensor<float>(input, new[] { 1, input.Length });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();
        return output.ToArray()[0];
    }
}