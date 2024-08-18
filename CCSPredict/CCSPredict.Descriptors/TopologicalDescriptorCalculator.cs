using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using GraphMolWrap;

namespace CCSPredict.Descriptors;

public class TopologicalDescriptorCalculator : IMoleculeDescriptorCalculator<float>
{
    public IEnumerable<string> SupportedDescriptors => new[]
    {
            //"WienerIndex",
            //"BalabanJ",
            //"BertzCT",
            //"Ipc",
            "HallKierAlpha",
            "Kappa1",
            "Kappa2",
            "Kappa3"
        };

    public async Task<Dictionary<string, float>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            try
            {
                var mol = RWMol.MolFromSmiles(molecule.Smiles);

                var descriptors = new Dictionary<string, float>
                {
                    ["HallKierAlpha"] = (float)RDKFuncs.calcHallKierAlpha(mol),
                    ["Kappa1"] = (float)RDKFuncs.calcKappa1(mol),
                    ["Kappa2"] = (float)RDKFuncs.calcKappa2(mol),
                    ["Kappa3"] = (float)RDKFuncs.calcKappa3(mol)
                };

                // Chi descriptors
                for (uint i = 0; i <= 4; i++)
                {
                    descriptors[$"Chi{i}v"] = (float)RDKFuncs.calcChiNn(mol, i);
                }

                return descriptors;
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                return new Dictionary<string, float>();
            }
        });
    }
}
