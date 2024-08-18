using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using com.epam.indigo;

namespace CCSPredict.Descriptors;

public class IndigoDescriptorCalculator : IMoleculeDescriptorCalculator<float>
{
    Indigo indigo = new Indigo();
    public IEnumerable<string> SupportedDescriptors => new[]
    {
            "MolecularWeight",
            "LogP",
            "TPSA",
            "HeavyAtomCount"
        };

    public async Task<Dictionary<string, float>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var indigoMolecule = indigo.loadMolecule(molecule.Smiles);
            var descriptors = new Dictionary<string, float>
            {
                ["MolecularWeight"] = (float)indigoMolecule.molecularWeight(),
                ["LogP"] = (float)indigoMolecule.logP(),
                ["TPSA"] = (float)indigoMolecule.tpsa(),
                ["HeavyAtomCount"] = indigoMolecule.countHeavyAtoms()
            };
            return descriptors;
        });
    }
}
