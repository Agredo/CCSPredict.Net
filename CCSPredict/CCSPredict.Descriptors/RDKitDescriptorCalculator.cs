using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using GraphMolWrap;

namespace CCSPredict.Descriptors;

public class RDKitDescriptorCalculator : IMoleculeDescriptorCalculator
{
    public IEnumerable<string> SupportedDescriptors => new[]
    {
            "ExactMolWt",
            "FractionCSP3",
            "NumRotatableBonds",
            "NumHDonors"
        };

    public async Task<Dictionary<string, double>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var mol = RWMol.MolFromSmiles(molecule.Smiles);
            var descriptors = new Dictionary<string, double>
            {
                ["ExactMolWt"] = RDKFuncs.calcExactMW(mol),
                ["FractionCSP3"] = RDKFuncs.calcFractionCSP3(mol),
                ["NumRotatableBonds"] = RDKFuncs.calcNumRotatableBonds(mol),
                //["NumHDonors"] = RDKFuncs.(mol)
            };
            return descriptors;
        });
    }
}
