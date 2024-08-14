using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CCSPredict.Models.DataModels;

public record Molecule
{
    public string Smiles { get; set; }
    public string InChI { get; set; }
    public Dictionary<string, double> Descriptors { get; set; } = new Dictionary<string, double>();

    public Molecule(string smiles, string inchi)
    {
        Smiles = smiles ?? throw new ArgumentNullException(nameof(smiles));
        InChI = inchi;
    }
}
