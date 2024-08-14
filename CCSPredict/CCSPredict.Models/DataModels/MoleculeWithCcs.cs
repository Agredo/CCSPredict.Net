namespace CCSPredict.Models.DataModels;

public class MoleculeWithCcs
{
    public string Smiles { get; set; }
    public string InChI { get; set; }
    public double CcsValue { get; set; }
}
