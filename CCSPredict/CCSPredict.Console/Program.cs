

using CCSPredict.Data;
using CCSPredict.Descriptors.Helper;
using CCSPredict.ML;
using System.Text;
using System.Text.RegularExpressions;

var dataProvider = new CsvCcsDataProvider("C:\\Users\\chris\\Documents\\training.csv", "C:\\Users\\chris\\Documents\\test_ccs.csv");
var predictor = new CcsPredictor(dataProvider);


await predictor.TrainAndEvaluateAsync();


Console.WriteLine("Enter SMILES to predict CCS (or 'exit' to quit):");
while (true)
{
    var input = Console.ReadLine();
    if (input.ToLower() == "exit") break;

    var smiles = "";

    if (StructureConverter.IsValidSmiles(input))
    {
        smiles = StructureConverter.ConvertToSmiles(input);
    }
    else
    {
        if (StructureConverter.IsValidInchi(input))
        {
            smiles = StructureConverter.ConvertToSmiles(input);
        }
        else
        {
            Console.WriteLine("Invalid input. Please enter a valid SMILES or InChI.");
        }
    }

    var prediction = await predictor.PredictCcsAsync(smiles);
    var svmPrediction = await predictor.PredictCcsSVMAsync(smiles);
    var randomForestPrediction = await predictor.PredictCcsRandomForestAsync(smiles);
    var neuralNetworkPrediction = await predictor.PredictCcsNeuralNetworkAsync(smiles);

    Console.WriteLine($"Predicted CCS (FastTreeTweedie): {prediction.PredictedCcs.Value} {prediction.PredictedCcs.Unit}");
    Console.WriteLine($"Predicted CCS (SVM): {svmPrediction.PredictedCcs.Value} {svmPrediction.PredictedCcs.Unit}");
    Console.WriteLine($"Predicted CCS (Random Forest): {randomForestPrediction.PredictedCcs.Value} {randomForestPrediction.PredictedCcs.Unit}");
    Console.WriteLine($"Predicted CCS (Neural Network): {neuralNetworkPrediction.PredictedCcs.Value} {neuralNetworkPrediction.PredictedCcs.Unit}");
    //Console.WriteLine($"Confidence: {prediction.Confidence}");

}


static void ConvertToCsv(string inputFilePath, string outputFilePath)
{
    try
    {
        // Lese den gesamten Inhalt der Datei
        string content = File.ReadAllText(inputFilePath);

        // Verwende Regex, um die Spalten korrekt zu trennen
        var regex = new Regex("(?:^|,)(\"(?:[^\"]+|\"\")*\"|[^,]*)", RegexOptions.Compiled);

        // Erstelle einen StringBuilder für die Ausgabe
        StringBuilder output = new StringBuilder();

        // Schreibe die Kopfzeile
        output.AppendLine("Adduct;MZ;Smiles;InChI;CcsValue");

        // Teile den Inhalt in Zeilen
        string[] lines = content.Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries);

        // Verarbeite jede Zeile (außer der Kopfzeile)
        for (int i = 1; i < lines.Length; i++)
        {
            var line = lines[i];
            var matches = regex.Matches(line);
            var columns = new List<string>();

            foreach (Match match in matches)
            {
                // Entferne die umschließenden Anführungszeichen und ersetze doppelte durch einzelne
                var value = match.Value.TrimStart(',');
                if (value.StartsWith("\"") && value.EndsWith("\""))
                {
                    value = value.Substring(1, value.Length - 2).Replace("\"\"", "\"");
                }
                columns.Add(value);
            }

            // Extrahiere die gewünschten Werte
            string adduct = columns[7];
            string mz = columns[6];
            string inchi = columns[3];
            string ccsValue = columns[11];

            // Füge die Werte zur Ausgabe hinzu
            output.AppendLine($"{adduct};{mz};;{inchi};{ccsValue}");
        }

        // Schreibe die Ausgabe in die Datei
        File.WriteAllText(outputFilePath, output.ToString());

        Console.WriteLine("Konvertierung abgeschlossen. Die Ausgabedatei wurde erstellt: " + outputFilePath);
    }
    catch (Exception ex)
    {
        Console.WriteLine("Ein Fehler ist aufgetreten: " + ex.Message);
    }
}
