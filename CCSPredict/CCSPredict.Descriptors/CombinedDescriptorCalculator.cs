using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;

namespace CCSPredict.Descriptors;

public class CombinedDescriptorCalculator : IMoleculeDescriptorCalculator
{
    private readonly List<IMoleculeDescriptorCalculator> calculators;

    public CombinedDescriptorCalculator()
    {
        calculators = new List<IMoleculeDescriptorCalculator>
            {
                new TopologicalDescriptorCalculator(),
                new GeometricDescriptorCalculator(),
                new ElectronicDescriptorCalculator(),
                new PhysicochemicalDescriptorCalculator(),
                new FingerprintCalculator()
            };
    }

    public IEnumerable<string> SupportedDescriptors =>
        calculators.SelectMany(c => c.SupportedDescriptors).Distinct();

    public async Task<Dictionary<string, double>> CalculateDescriptorsAsync(Molecule molecule)
    {
        var allDescriptors = new Dictionary<string, double>();

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
}
