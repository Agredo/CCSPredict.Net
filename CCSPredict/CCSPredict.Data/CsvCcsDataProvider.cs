using CCSPredict.Models.DataModels;
using CsvHelper;
using CsvHelper.Configuration;
using System.Formats.Asn1;
using System.Globalization;

namespace CCSPredict.Data;

public class CsvCcsDataProvider : ICcsDataProvider
{
    private readonly string trainingDataPath;
    private readonly string testDataPath;

    public CsvCcsDataProvider(string trainingDataPath, string testDataPath)
    {
        this.trainingDataPath = trainingDataPath;
        this.testDataPath = testDataPath;
    }

    public async Task<IEnumerable<MoleculeWithCcs>> GetTrainingDataAsync()
    {
        return await ReadCsvFileAsync(trainingDataPath);
    }

    public async Task<IEnumerable<MoleculeWithCcs>> GetTestDataAsync()
    {
        return await ReadCsvFileAsync(testDataPath);
    }

    private async Task<IEnumerable<MoleculeWithCcs>> ReadCsvFileAsync(string filePath)
    {
        using var reader = new StreamReader(filePath);
        using var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            Delimiter = ";",
            HeaderValidated = null,
            MissingFieldFound = null
        });

        var records = new List<MoleculeWithCcs>();
        await foreach (var record in csv.GetRecordsAsync<MoleculeWithCcs>())
        {
            records.Add(record);
        }

        return records;
    }
}