using CCSPredict.Models.DataModels;

namespace CCSPredict.Core.Interfaces;

public interface IMoleculeDescriptorCalculator
{
    Task<Dictionary<string, double>> CalculateDescriptorsAsync(Molecule molecule);
    IEnumerable<string> SupportedDescriptors { get; }
}
