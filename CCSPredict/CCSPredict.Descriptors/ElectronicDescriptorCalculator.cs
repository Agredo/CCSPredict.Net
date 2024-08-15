using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using GraphMolWrap;

namespace CCSPredict.Descriptors;

public class ElectronicDescriptorCalculator : IMoleculeDescriptorCalculator
{
    public IEnumerable<string> SupportedDescriptors => new[]
    {
            "MaxPartialCharge",
            "MinPartialCharge",
            "MeanPartialCharge",
            "NumRadicalElectrons"
        };

    public async Task<Dictionary<string, float>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var mol = RWMol.MolFromSmiles(molecule.Smiles);
            var descriptors = new Dictionary<string, float>();
            var charges = new float[mol.getNumAtoms()];
            for (int i = 0; i < mol.getNumAtoms(); i++)
            {
                Atom atom = mol.getAtomWithIdx((uint)i);
                if (atom.hasProp("_GasteigerCharge"))
                {
                    string gasteigerCharge = atom.getProp("_GasteigerCharge");

                    charges[i] = (float)Convert.ToDouble(gasteigerCharge);

                    descriptors = new Dictionary<string, float>
                    {
                        ["MaxPartialCharge"] = charges.Max(),
                        ["MinPartialCharge"] = charges.Min(),
                        ["MeanPartialCharge"] = charges.Average()
                    };
                }
            }


            return descriptors;
        });
    }
}