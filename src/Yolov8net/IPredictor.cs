using System.Drawing;

namespace Yolov8Net
{
    public interface IPredictor
        : IDisposable
    {
        string? InputColumnName { get; }
        string? OutputColumnName { get; }

        int ModelInputHeight { get; }
        int ModelInputWidth { get; }

        int ModelOutputDimensions { get; }

        Prediction[] Predict(Image img);
    }
}
