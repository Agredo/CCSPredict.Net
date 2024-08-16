using com.epam.indigo;

namespace CCSPredict.Descriptors.Helper
{
    public static class StructureConverter
    {
        public static string InchiToSmiles(string inchi)
        {
            Indigo indigo = new Indigo();
            IndigoObject mol = indigo.loadMolecule(inchi);

            if (mol == null)
            {
                throw new Exception("Ungültiger InChI-String");
            }

            return mol.canonicalSmiles();
        }
    }
}
