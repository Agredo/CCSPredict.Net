using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using com.epam.indigo;
using GraphMolWrap;

namespace CCSPredict.Descriptors;

public class FingerprintCalculator : IMoleculeDescriptorCalculator<List<float>>
{
    private readonly Indigo indigo = new Indigo();

    public IEnumerable<string> SupportedDescriptors => new[]
    {
        "MorganFingerprint",
        "AtomPairFingerprint",
        "TopologicalTorsionFingerprint",
        "MACCSFingerprint"
    };

    public async Task<Dictionary<string, List<float>>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var rdkitMol = RWMol.MolFromSmiles(molecule.Smiles);

            var descriptors = new Dictionary<string, List<float>>();

            // Morgan (ECFP) Fingerprint
            var morganFp = RDKFuncs.getMorganFingerprintAsBitVect(rdkitMol, 3, 2048);
            descriptors["MorganFingerprint"] = CalculateFingerprint(morganFp);

            //SparseIntVect32 atomPairFingerprint = RDKFuncs.getAtomPairFingerprint(rdkitMol);
            //descriptors["AtomPairFingerprint"] = CalculateFingerprint(atomPairFingerprint);

            //SparseIntVect64 topologicalFingerprint = RDKFuncs.getTopologicalTorsionFingerprint(rdkitMol);
            //descriptors["TopologicalTorsionFingerprint"] = CalculateFingerprint(topologicalFingerprint);

            ExplicitBitVect maccsFingerprint = RDKFuncs.MACCSFingerprintMol(rdkitMol);
            descriptors["MACCSFingerprint"] = CalculateFingerprint(maccsFingerprint);

            return descriptors;
        });
    }

    private float BitVectToDouble(ExplicitBitVect bv)
    {
        return bv.getNumOnBits() / bv.getNumBits();
    }

    private new List<float> CalculateFingerprint(ExplicitBitVect fingerprint)
    {
        var bits = new List<float>();

        for (int i = 0; i < fingerprint.getNumBits(); i++)
        {
            bits.Add(fingerprint.getBit((uint)i) ? 1:0);
        }
        return bits;
    }

    private new List<float> CalculateFingerprint(SparseIntVect32 fingerprint)
    {
        var bits = new List<float>();

        for (int i = 0; i < fingerprint.getLength(); i++)
        {
            bits.Add(fingerprint.getVal(i));
        }
        return bits;
    }

    private new List<float> CalculateFingerprint(SparseIntVect64 fingerprint)
    {
        var bits = new List<float>();

        for (int i = 0; i < fingerprint.getLength(); i++)
        {
            bits.Add(fingerprint.getVal(i));
        }
        return bits;
    }


    private float SparseIntVectToDouble(SparseIntVect32 siv)
    {
        return siv.getNonzero().Count / siv.getLength();
    }
    private float SparseIntVectToDouble(SparseIntVect64 siv)
    {
        return siv.getNonzero().Count / siv.getLength();
    }

    private float FingerprintToDouble(IndigoObject fp)
    {
        return fp.countBits() / fp.toString().Length;
    }
}