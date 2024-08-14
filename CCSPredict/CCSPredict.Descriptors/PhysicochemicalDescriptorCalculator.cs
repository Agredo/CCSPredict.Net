using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using com.epam.indigo;
using GraphMolWrap;

namespace CCSPredict.Descriptors;

public class PhysicochemicalDescriptorCalculator : IMoleculeDescriptorCalculator
{
    private readonly Indigo indigo = new Indigo();

    public IEnumerable<string> SupportedDescriptors => new[]
    {
            "MolWt",
            "ExactMolWt",
            "LogP",
            "NumHDonors",
            "NumHAcceptors",
            "NumRotatableBonds",
            "NumHeavyAtoms",
            "FractionCSP3",
            "IndigoLogP",
            "IndigoPSA"
        };

    public async Task<Dictionary<string, double>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var rdkitMol = RWMol.MolFromSmiles(molecule.Smiles);
            var indigoMol = indigo.loadMolecule(molecule.Smiles);

            var descriptors = new Dictionary<string, double>
            {
                //["MolWt"] = RDKFuncs.calcMolWt(rdkitMol),
                ["ExactMolWt"] = RDKFuncs.calcExactMW(rdkitMol),
                //["LogP"] = RDKFuncs.calcCrippenDescriptors(rdkitMol)[0],
                //["NumHDonors"] = RDKFuncs.calcNumHDonors(rdkitMol),
                //["NumHAcceptors"] = RDKFuncs.calcNumHAcceptors(rdkitMol),
                ["NumRotatableBonds"] = RDKFuncs.calcNumRotatableBonds(rdkitMol),
                ["NumHeavyAtoms"] = rdkitMol.getNumHeavyAtoms(),
                ["FractionCSP3"] = RDKFuncs.calcFractionCSP3(rdkitMol),
                ["IndigoLogP"] = indigoMol.logP(),
                //["IndigoPSA"] = indigoMol.polarSurfaceArea()
            };
            return descriptors;
        });
    }
}
