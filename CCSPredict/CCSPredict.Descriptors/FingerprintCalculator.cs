using CCSPredict.Core.Interfaces;
using CCSPredict.Models.DataModels;
using com.epam.indigo;
using GraphMolWrap;

namespace CCSPredict.Descriptors;

public class FingerprintCalculator : IMoleculeDescriptorCalculator
{
    private readonly Indigo indigo = new Indigo();

    public IEnumerable<string> SupportedDescriptors => new[]
    {
            "MorganFingerprint",
            "AtomPairFingerprint",
            "TopologicalTorsionFingerprint",
            "IndigoFingerprint"
        };

    public async Task<Dictionary<string, double>> CalculateDescriptorsAsync(Molecule molecule)
    {
        return await Task.Run(() =>
        {
            var rdkitMol = RWMol.MolFromSmiles(molecule.Smiles);
            var indigoMol = indigo.loadMolecule(molecule.Smiles);

            var descriptors = new Dictionary<string, double>();

            // Morgan (ECFP) Fingerprint
            var morganFp = RDKFuncs.getMorganFingerprintAsBitVect(rdkitMol, 2, 2048);
            descriptors["MorganFingerprint"] = BitVectToDouble(morganFp);

            // Atom Pair Fingerprint
            var atomPairFp = RDKFuncs.getAtomPairFingerprint(rdkitMol);
            descriptors["AtomPairFingerprint"] = SparseIntVectToDouble(atomPairFp) ;

            // Topological Torsion Fingerprint
            var topologicalTorsionFp = RDKFuncs.getTopologicalTorsionFingerprint(rdkitMol);
            descriptors["TopologicalTorsionFingerprint"] = SparseIntVectToDouble(topologicalTorsionFp);

            // Indigo Fingerprint
            var indigoFp = indigoMol.fingerprint("sim");
            descriptors["IndigoFingerprint"] = FingerprintToDouble(indigoFp);

            return descriptors;
        });
    }

    private double BitVectToDouble(ExplicitBitVect bv)
    {
        return (double)bv.getNumOnBits() / bv.getNumBits();
    }

    private double SparseIntVectToDouble(SparseIntVect32 siv)
    {
        return siv.getNonzero().Count / (double)siv.getLength();
    }
    private double SparseIntVectToDouble(SparseIntVect64 siv)
    {
        return siv.getNonzero().Count / (double)siv.getLength();
    }

    private double FingerprintToDouble(IndigoObject fp)
    {
        return fp.countBits() / (double)fp.toString().Length;
    }
}