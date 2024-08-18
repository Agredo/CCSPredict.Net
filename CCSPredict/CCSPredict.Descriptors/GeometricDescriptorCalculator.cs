using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using com.epam.indigo;
using GraphMolWrap;

namespace CCSPredict.Descriptors;

public class GeometricDescriptorCalculator : IMoleculeDescriptorCalculator<float>
{
    private readonly Indigo indigo = new Indigo();

    public IEnumerable<string> SupportedDescriptors => new[]
    {
            //"MolecularVolume",
            "TPSA",
            "LabuteASA",
            //"IndigoVolume",
            //"IndigoSurfaceArea"
        };

    public async Task<Dictionary<string, float>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var rdkitMol = RWMol.MolFromSmiles(molecule.Smiles);

            var indigoMol = indigo.loadMolecule(molecule.Smiles);
            indigoMol.aromatize();
            indigoMol.layout();

            var descriptors = new Dictionary<string, float>
            {
                ["TPSA"] = (float)RDKFuncs.calcTPSA(rdkitMol),
                ["LabuteASA"] = (float)RDKFuncs.calcLabuteASA(rdkitMol)
            };
            return descriptors;
        });
    }
}
