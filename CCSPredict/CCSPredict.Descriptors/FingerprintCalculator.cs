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
        };

    public async Task<Dictionary<string, List<float>>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var rdkitMol = RWMol.MolFromSmiles(molecule.Smiles);

            var descriptors = new Dictionary<string, List<float>>();

            // Morgan (ECFP) Fingerprint
            var morganFp = RDKFuncs.getMorganFingerprintAsBitVect(rdkitMol, 2, 2048);
            descriptors["MorganFingerprint"] = CalculateMorganFingerprint(morganFp);

            //// Atom Pair Fingerprint
            //var atomPairFp = RDKFuncs.getAtomPairFingerprint(rdkitMol);
            //descriptors["AtomPairFingerprint"] = SparseIntVectToDouble(atomPairFp);

            //// Topological Torsion Fingerprint
            //var topologicalTorsionFp = RDKFuncs.getTopologicalTorsionFingerprint(rdkitMol);
            //descriptors["TopologicalTorsionFingerprint"] = SparseIntVectToDouble(topologicalTorsionFp);

            //// Indigo Fingerprint
            //var indigoFp = indigoMol.fingerprint("sim");
            //descriptors["IndigoFingerprint"] = FingerprintToDouble(indigoFp);



            return descriptors;
        });
    }

    private float BitVectToDouble(ExplicitBitVect bv)
    {
        return bv.getNumOnBits() / bv.getNumBits();
    }

    private new List<float> CalculateMorganFingerprint(ExplicitBitVect fingerprint)
    {
        var bits = new List<float>();

        for (int i = 0; i < fingerprint.getNumBits(); i++)
        {
            bits.Add(fingerprint.getBit((uint)i) ? 1:0);
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