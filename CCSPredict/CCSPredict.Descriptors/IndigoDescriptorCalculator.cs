using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using com.epam.indigo;

namespace CCSPredict.Descriptors;

public class IndigoDescriptorCalculator : IMoleculeDescriptorCalculator
{
    private readonly Indigo indigo = new Indigo();

    public IEnumerable<string> SupportedDescriptors => new[]
    {
            "MolecularWeight",
            "LogP",
            "TPSA",
            "HeavyAtomCount"
        };

    public async Task<Dictionary<string, double>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var indigoMolecule = indigo.loadMolecule(molecule.Smiles);
            var descriptors = new Dictionary<string, double>
            {
                ["MolecularWeight"] = indigoMolecule.molecularWeight(),
                ["LogP"] = indigoMolecule.logP(),
                ["TPSA"] = indigoMolecule.tpsa(),
                ["HeavyAtomCount"] = indigoMolecule.countHeavyAtoms()
            };
            return descriptors;
        });
    }
}
