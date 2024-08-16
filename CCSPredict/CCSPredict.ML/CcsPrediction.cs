using Microsoft.ML.Data;

namespace CCSPredict.ML;

public class CcsPrediction
{
    [ColumnName("Score")]
    public float CcsValue;
}