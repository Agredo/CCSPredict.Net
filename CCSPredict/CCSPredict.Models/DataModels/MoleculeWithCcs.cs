namespace CCSPredict.Models.DataModels;

public class MoleculeWithCcs
{
    public string Smiles { get; set; }
    public string InChI { get; set; }
    public double MZ { get; set; }
    public string Adduct { get; set; }
    public double CcsValue { get; set; }
}
