using CCSPredict.Data;
using CCSPredict.Descriptors;
using CCSPredict.Models.DataModels;
using Microsoft.ML;

namespace CCSPredict.ML.PredictionModels
{
    public interface IPredictionModel
    {
        IDataView data { get; set; }

        MLContext mlContext { get; set; }
        ITransformer model { get; set; }
        CombinedFloatDescriptorCalculator descriptorCalculator { get; set; }



        Task<ModelMetrics> EvaluateAsync();
        void LoadModel(string filePath);
        Task<CcsPredictionResult> PredictAsync(Molecule molecule);
        void SaveModel(string filePath);
        void SaveOnnxModel(string filePath = "./model.onnx");
        Task TrainAsync(double testFraction);
        double CalculateConfidence(CcsPrediction prediction);
    }
}