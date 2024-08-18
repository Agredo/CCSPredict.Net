using CCSPredict.Models.DataModels;

namespace CCSPredict.Core.Interfaces;

public interface IMoleculeDescriptorCalculator<T>
{
    Task<Dictionary<string, T>> CalculateDescriptorsAsync(Molecule molecule);
    IEnumerable<string> SupportedDescriptors { get; }
}
