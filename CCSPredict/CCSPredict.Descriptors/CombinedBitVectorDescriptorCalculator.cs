using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;

namespace CCSPredict.Descriptors;

public class CombinedBitVectorDescriptorCalculator : IMoleculeDescriptorCalculator<List<float>>
{
    private readonly List<IMoleculeDescriptorCalculator<List<float>>> calculators;

    public CombinedBitVectorDescriptorCalculator()
    {
        calculators = new List<IMoleculeDescriptorCalculator<List<float>>>
            {
                new FingerprintCalculator()
            };
    }

    public IEnumerable<string> SupportedDescriptors =>
        calculators.SelectMany(c => c.SupportedDescriptors).Distinct();

    public async Task<Dictionary<string, List<float>>> CalculateDescriptorsAsync(Molecule molecule)
    {
        var allDescriptors = new Dictionary<string, List<float>>();

        foreach (var calculator in calculators)
        {
            var descriptors = await calculator.CalculateDescriptorsAsync(molecule);
            foreach (var kvp in descriptors)
            {
                allDescriptors[kvp.Key] = kvp.Value;
            }
        }

        return allDescriptors;
    }
    public Dictionary<string, List<float>> CalculateDescriptors(Molecule molecule)
    {
        var allDescriptors = new Dictionary<string, List<float>>();

        foreach (var calculator in calculators)
        {
            var descriptors =  calculator.CalculateDescriptorsAsync(molecule).Result;
            foreach (var kvp in descriptors)
            {
                allDescriptors[kvp.Key] = kvp.Value;
            }
        }

        return allDescriptors;
    }
}
