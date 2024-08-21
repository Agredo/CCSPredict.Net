using CCSPredict.Descriptors;
using CCSPredict.ML;
using CCSPredict.Models.DataModels;
using GraphMolWrap;
using Microsoft.ML.Data;
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

            CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator = new CombinedBitVectorDescriptorCalculator();
            CombinedFloatDescriptorCalculator descriptorCalculator = new CombinedFloatDescriptorCalculator();

            RWMol mol = RWMol.MolFromSmiles(input);
            var molecule = new Molecule(mol.MolToSmiles(), "");

            var descriptors = descriptorCalculator.CalculateDescriptors(molecule);
            var bitVectorDictionary = bitVectorDescriptorCalculator.CalculateDescriptors(molecule);

            MoleculeData moleculeData = GetMoleculeData(descriptors, bitVectorDictionary);

            var descriptorsArray = CombineDescriptorsAndFingerprints(moleculeData);

            var fastTreePrediction = fastTreeModel.Predict(descriptorsArray);
            var svmPrediction = svmModel.Predict(descriptorsArray);
            var randomForestPrediction = randomForestModel.Predict(descriptorsArray);
            var neuralNetworkPrediction = neuralNetworkModel.Predict(descriptorsArray);

            Console.WriteLine($"Predicted CCS (FastTree): {fastTreePrediction} Å²");
            Console.WriteLine($"Predicted CCS (SVM): {svmPrediction} Å²");
            Console.WriteLine($"Predicted CCS (Random Forest): {randomForestPrediction} Å²");
            Console.WriteLine($"Predicted CCS (Neural Network): {neuralNetworkPrediction} Å²");
        }

        static float[] CombineDescriptorsAndFingerprints(MoleculeData moleculeData)
        {
            var scalarDescriptors = new List<float>
        {
            moleculeData.HallKierAlpha,
            moleculeData.Kappa1,
            moleculeData.Kappa2,
            moleculeData.Kappa3,
            moleculeData.Chi0v,
            moleculeData.Chi1v,
            moleculeData.Chi2v,
            moleculeData.Chi3v,
            moleculeData.TPSA,
            moleculeData.LabuteASA,
            moleculeData.MolecularWeight,
            moleculeData.NumHeavyAtoms,
            moleculeData.FractionCSP3
        };

            // Konvertieren Sie die VBuffer<float> Fingerprints in reguläre float-Arrays
            var morganFingerprint = ConvertVBufferToArray(moleculeData.MorganFingerprint);
            var maccsFingerprint = ConvertVBufferToArray(moleculeData.MACCSFingerprint);

            // Kombinieren Sie alle Deskriptoren und Fingerprints in ein einziges Array
            return scalarDescriptors
                .Concat(morganFingerprint)
                .Concat(maccsFingerprint)
                .ToArray();
        }

        static float[] ConvertVBufferToArray(VBuffer<float> vbuffer)
        {
            if (vbuffer.IsDense)
            {
                return vbuffer.DenseValues().ToArray();
            }
            else
            {
                var result = new float[vbuffer.Length];
                var values = vbuffer.GetValues();
                var indices = vbuffer.GetIndices();
                for (int i = 0; i < values.Length; i++)
                {
                    result[indices[i]] = values[i];
                }
                return result;
            }
        }
    }

    public static MoleculeData GetMoleculeData(Dictionary<string, float> descriptors, Dictionary<string, List<float>> bitVectorDictionary)
    {

        MoleculeData molecueData = new MoleculeData
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
        };

        // Calculate fingerprints
        molecueData.MorganFingerprint = new VBuffer<float>(bitVectorDictionary["MorganFingerprint"].Count, bitVectorDictionary["MorganFingerprint"].ToArray());
        //molecueData.AtomPairFingerprint = CalculateAtomPairFingerprint(mol, 2048);
        //molecueData.TopologicalTorsionFingerprint = CalculateTopologicalTorsionFingerprint(mol, 2048);
        molecueData.MACCSFingerprint = new VBuffer<float>(bitVectorDictionary["MACCSFingerprint"].Count, bitVectorDictionary["MACCSFingerprint"].ToArray());

        return molecueData;
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