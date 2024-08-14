using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using com.epam.indigo;
using GraphMolWrap;

namespace CCSPredict.Descriptors;

public class TopologicalDescriptorCalculator : IMoleculeDescriptorCalculator
{
    public IEnumerable<string> SupportedDescriptors => new[]
    {
            "WienerIndex",
            "BalabanJ",
            "BertzCT",
            "Ipc",
            "HallKierAlpha",
            "Kappa1",
            "Kappa2",
            "Kappa3"
        };

    public async Task<Dictionary<string, double>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var mol = RWMol.MolFromSmiles(molecule.Smiles);

            var descriptors = new Dictionary<string, double>
            {
                ["HallKierAlpha"] = RDKFuncs.calcHallKierAlpha(mol),
                ["Kappa1"] = RDKFuncs.calcKappa1(mol),
                ["Kappa2"] = RDKFuncs.calcKappa2(mol),
                ["Kappa3"] = RDKFuncs.calcKappa3(mol)
            };

            // Chi descriptors
            for (uint i = 0; i <= 4; i++)
            {
                descriptors[$"Chi{i}v"] = RDKFuncs.calcChiNn(mol, i);
            }

            return descriptors;
        });
    }
}
