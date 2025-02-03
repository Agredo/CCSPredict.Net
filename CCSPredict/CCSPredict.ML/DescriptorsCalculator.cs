using CCSPredict.Descriptors;
using CCSPredict.Models.DataModels;
using GraphMolWrap;
using Microsoft.ML.Data;
using System;

namespace CCSPredict.ML;


/// <summary>
/// Class responsible for calculating descriptors for molecules.
/// </summary>
public class DescriptorsCalculator
{
    private readonly CombinedFloatDescriptorCalculator descriptorCalculator;
    private readonly CombinedBitVectorDescriptorCalculator bitVectorDescriptorCalculator;

    /// <summary>
    /// Initializes a new instance of the <see cref="DescriptorsCalculator"/> class.
    /// </summary>
    public DescriptorsCalculator()
    {
        this.descriptorCalculator = new CombinedFloatDescriptorCalculator();
        this.bitVectorDescriptorCalculator = new CombinedBitVectorDescriptorCalculator();
    }

    /// <summary>
    /// Calculates the descriptors asynchronously for the given SMILES and InChI.
    /// </summary>
    /// <param name="smiles">The SMILES representation of the molecule.</param>
    /// <param name="inChI">The InChI representation of the molecule.</param>
    /// <returns>The calculated MoleculeData.</returns>
    public async Task<MoleculeData> CalculateDescriptorsAsync(string smiles, string inChI)
    {
        var molecule = new Molecule(smiles, inChI);

        if (!string.IsNullOrEmpty(smiles))
        {
            return await CalculateDescriptorsAsync(molecule);
        }
        else if (string.IsNullOrEmpty(molecule.Smiles) && !string.IsNullOrEmpty(molecule.InChI))
        {
            try
            {
                ExtraInchiReturnValues extra = new ExtraInchiReturnValues();
                RWMol mol = RDKFuncs.InchiToMol(molecule.InChI, extra);

                molecule.Smiles = mol.MolToSmiles();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to convert InChI to SMILES for molecule {molecule.InChI}: {ex.Message}");
                throw;
            }
        }

        return await CalculateDescriptorsAsync(molecule);
    }

    /// <summary>
    /// Calculates the descriptors asynchronously for the given Molecule.
    /// </summary>
    /// <param name="molecule">The Molecule object.</param>
    /// <returns>The calculated MoleculeData.</returns>
    public async Task<MoleculeData> CalculateDescriptorsAsync(Molecule molecule)
    {
        var descriptorCalculator = new CombinedFloatDescriptorCalculator();
        var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(molecule);
        var bitVectorDescriptors = await bitVectorDescriptorCalculator.CalculateDescriptorsAsync(molecule);

        return BuildMolecule(descriptors, bitVectorDescriptors, 0f);
    }

    /// <summary>
    /// Calculates the descriptors asynchronously for the given MoleculeWithCcs.
    /// </summary>
    /// <param name="molecule">The MoleculeWithCcs object.</param>
    /// <returns>The calculated MoleculeData.</returns>
    public async Task<MoleculeData> CalculateDescriptorsAsync(MoleculeWithCcs molecule)
    {
        var descriptors = await descriptorCalculator.CalculateDescriptorsAsync(new Molecule(molecule.Smiles, molecule.InChI));
        var bitVectorDescriptors = await bitVectorDescriptorCalculator.CalculateDescriptorsAsync(new Molecule(molecule.Smiles, molecule.InChI));
        return BuildMolecule(descriptors, bitVectorDescriptors, (float)molecule.CcsValue);
    }

    private MoleculeData BuildMolecule(Dictionary<string, float> descriptors, Dictionary<string, List<float>> bitVectorDescriptors, float ccsValue)
    {
        return new MoleculeData
        {
            HallKierAlpha = descriptors["HallKierAlpha"],
            Kappa1 = descriptors["Kappa1"],
            Kappa2 = descriptors["Kappa2"],
            Kappa3 = descriptors["Kappa3"],
            Chi0v = descriptors["Chi0v"],
            Chi1v = descriptors["Chi1v"],
            Chi2v = descriptors["Chi2v"],
            Chi3v = descriptors["Chi3v"],
            TPSA = descriptors["TPSA"],
            LabuteASA = descriptors["LabuteASA"],
            MolecularWeight = descriptors["ExactMolWt"],
            NumHeavyAtoms = descriptors["NumHeavyAtoms"],
            FractionCSP3 = descriptors["FractionCSP3"],
            MorganFingerprint = new VBuffer<float>(bitVectorDescriptors["MorganFingerprint"].Count, bitVectorDescriptors["MorganFingerprint"].ToArray()),
            MACCSFingerprint = new VBuffer<float>(bitVectorDescriptors["MACCSFingerprint"].Count, bitVectorDescriptors["MACCSFingerprint"].ToArray()),

            CcsValue = ccsValue
        };
    }
}
