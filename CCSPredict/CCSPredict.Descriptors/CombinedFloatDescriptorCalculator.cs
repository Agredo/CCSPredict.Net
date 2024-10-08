﻿using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;

namespace CCSPredict.Descriptors;

public class CombinedFloatDescriptorCalculator : IMoleculeDescriptorCalculator<float>
{
    private readonly List<IMoleculeDescriptorCalculator<float>> calculators;

    public CombinedFloatDescriptorCalculator()
    {
        calculators = new List<IMoleculeDescriptorCalculator<float>>
            {
                new TopologicalDescriptorCalculator(),
                new GeometricDescriptorCalculator(),
                //new ElectronicDescriptorCalculator(),
                new PhysicochemicalDescriptorCalculator()
            };
    }

    public IEnumerable<string> SupportedDescriptors =>
        calculators.SelectMany(c => c.SupportedDescriptors).Distinct();

    public async Task<Dictionary<string, float>> CalculateDescriptorsAsync(Molecule molecule)
    {
        var allDescriptors = new Dictionary<string, float>();

        foreach (var calculator in calculators)
        {
            var descriptors = await calculator.CalculateDescriptorsAsync(molecule);
            foreach (var kvp in descriptors)
            {
                allDescriptors[kvp.Key] = (float)kvp.Value;
            }
        }

        return allDescriptors;
    }
    public Dictionary<string, float> CalculateDescriptors(Molecule molecule)
    {
        var allDescriptors = new Dictionary<string, float>();

        foreach (var calculator in calculators)
        {
            var descriptors =  calculator.CalculateDescriptorsAsync(molecule).Result;
            foreach (var kvp in descriptors)
            {
                allDescriptors[kvp.Key] = (float)kvp.Value;
            }
        }

        return allDescriptors;
    }
}
