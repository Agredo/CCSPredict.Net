using CCSPredict.Models.DataModels;
using GraphMolWrap;

namespace CCSPredict.ML.Helper;

public static class StructureConverter
{
    public static string ConvertToSmiles(string inchi)
    {
        try
        {
            ExtraInchiReturnValues extra = new ExtraInchiReturnValues();
            RWMol mol = RDKFuncs.InchiToMol(inchi, extra);

            return mol.MolToSmiles();
        }
        catch(Exception ex)
        {
            Console.WriteLine($"Failed to convert InChI to SMILES. PLease insert an SMILES");
            return string.Empty;
        }
    }

    public static bool IsValidInchi(string inchi)
    {
        try
        {
            ExtraInchiReturnValues extra = new ExtraInchiReturnValues();
            RWMol mol = RDKFuncs.InchiToMol(inchi, extra);

            return true;
        }
        catch (Exception ex)
        {
            return false;
        }
    }

    public static bool IsValidSmiles(string smiles)
    {
        try
        {
            RWMol mol = RWMol.MolFromSmiles(smiles);

            return true;
        }
        catch (Exception ex)
        {
            return false;
        }
    }
}
