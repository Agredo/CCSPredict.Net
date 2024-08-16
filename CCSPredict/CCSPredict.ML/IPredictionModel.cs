using CCSPredict.Data;
using CCSPredict.Descriptors;
using CCSPredict.Models.DataModels;
using Microsoft.ML;

namespace CCSPredict.ML
{
    public interface IPredictionModel
    {
        IDataView data { get; set; }

        MLContext mlContext { get; set; }
        ITransformer model { get; set; }
        ICcsDataProvider dataProvider { get; set; }
        CombinedDescriptorCalculator descriptorCalculator { get; set; }



        Task<ModelMetrics> EvaluateAsync();
        void LoadModel(string filePath);
        Task<IEnumerable<MoleculeData>> PrepareDataAsync(IEnumerable<MoleculeWithCcs> molecules);
        Task<CcsPredictionResult> PredictAsync(Molecule molecule);
        void SaveModel(string filePath);
        void SaveOnnxModel(string filePath = "./model.onnx");
        Task TrainAsync();
        double CalculateConfidence(CcsPrediction prediction);
    }
}