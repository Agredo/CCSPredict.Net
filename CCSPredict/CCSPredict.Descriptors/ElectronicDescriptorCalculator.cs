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

    public async Task<Dictionary<string, double>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var mol = RWMol.MolFromSmiles(molecule.Smiles);

            var charges = new double[mol.getNumAtoms()];
            for (int i = 0; i < mol.getNumAtoms(); i++)
            {
                charges[i] = Convert.ToDouble(mol.getAtomWithIdx((uint)i).getProp("_GasteigerCharge"));
            }

            var descriptors = new Dictionary<string, double>
            {
                ["MaxPartialCharge"] = charges.Max(),
                ["MinPartialCharge"] = charges.Min(),
                ["MeanPartialCharge"] = charges.Average()
            };
            return descriptors;
        });
    }
}