using CCSPredict.Models.DataModels;

namespace CCSPredict.Data;

public interface ICcsDataProvider
{
    Task<IEnumerable<MoleculeWithCcs>> GetTrainingDataAsync();
    Task<IEnumerable<MoleculeWithCcs>> GetTestDataAsync();
}
