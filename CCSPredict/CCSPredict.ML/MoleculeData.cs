using Microsoft.ML.Data;

namespace CCSPredict.ML;

public class MoleculeData
{
    [ColumnName("HallKierAlpha")]
    public float HallKierAlpha { get; set; }

    [ColumnName("Kappa1")]
    public float Kappa1 { get; set; }

    [ColumnName("Kappa2")]
    public float Kappa2 { get; set; }

    [ColumnName("Kappa3")]
    public float Kappa3 { get; set; }

    [ColumnName("Chi0v")]
    public float Chi0v { get; set; }

    [ColumnName("Chi1v")]
    public float Chi1v { get; set; }

    [ColumnName("Chi2v")]
    public float Chi2v { get; set; }

    [ColumnName("Chi3v")]
    public float Chi3v { get; set; }

    [ColumnName("MolecularWeight")]
    public float MolecularWeight { get; set; }

    [ColumnName("TPSA")]
    public float TPSA { get; set; }

    [ColumnName("LogP")]
    public float LogP { get; set; }

    [ColumnName("LabuteASA")]
    public float LabuteASA { get; set; }

    [ColumnName("NumRadicalElectrons")]
    public float NumRadicalElectrons { get; set; }

    [ColumnName("MaxPartialCharge")]
    public float MaxPartialCharge { get; set; }

    [ColumnName("MinPartialCharge")]
    public float MinPartialCharge { get; set; }

    [ColumnName("MeanPartialCharge")]
    public float MeanPartialCharge { get; set; }

    [ColumnName("FractionCSP3")]
    public float FractionCSP3 { get; set; }

    [ColumnName("ExactMolWt")]
    public float ExactMolWt { get; set; }

    [ColumnName("NumRotatableBonds")]
    public float NumRotatableBonds { get; set; }

    [ColumnName("NumHeavyAtoms")]
    public float NumHeavyAtoms { get; set; }

    [ColumnName("MorganFingerprint")]
    [VectorType(2048)]
    public VBuffer<float> MorganFingerprint { get; set; }

    [ColumnName("AtomPairFingerprint")]
    public float AtomPairFingerprint { get; set; }

    [ColumnName("TopologicalTorsionFingerprint")]
    public float TopologicalTorsionFingerprint { get; set; }

    [ColumnName("CcsValue")]
    public float CcsValue { get; set; }


}
