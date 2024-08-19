using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        string basePath = "C:\\Projects\\CCSPredict.Net\\CCSPredict\\CCSPredict.Console\\bin\\x64\\Debug\\net8.0\\";
        var fastTreeModel = new OnnxModel(basePath + "fast_model.onnx");
        var svmModel = new OnnxModel(basePath + "svm_model.onnx");
        var randomForestModel = new OnnxModel(basePath + "randomForest_model.onnx");
        var neuralNetworkModel = new OnnxModel(basePath + "neuronal_model.onnx");

        Console.WriteLine("Enter SMILES to predict CCS (or 'exit' to quit):");
        while (true)
        {
            var input = Console.ReadLine();
            if (input.ToLower() == "exit") break;

            // For simplicity, we're assuming the input is always a valid SMILES string
            var descriptors = CalculateDescriptors(input);

            var fastTreePrediction = fastTreeModel.Predict(descriptors);
            var svmPrediction = svmModel.Predict(descriptors);
            var randomForestPrediction = randomForestModel.Predict(descriptors);
            var neuralNetworkPrediction = neuralNetworkModel.Predict(descriptors);

            Console.WriteLine($"Predicted CCS (FastTree): {fastTreePrediction} Å²");
            Console.WriteLine($"Predicted CCS (SVM): {svmPrediction} Å²");
            Console.WriteLine($"Predicted CCS (Random Forest): {randomForestPrediction} Å²");
            Console.WriteLine($"Predicted CCS (Neural Network): {neuralNetworkPrediction} Å²");
        }
    }

    static float[] CalculateDescriptors(string smiles)
    {
        // This is a placeholder. In a real application, you would calculate
        // the actual descriptors here using RDKit or another cheminformatics library.
        // The descriptor order should match what the models expect.
        return new float[20]; // Adjust the size based on your actual descriptor count
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