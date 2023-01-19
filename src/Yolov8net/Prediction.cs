using System.Drawing;

namespace Yolonet
{
    public class Prediction
    {
        public Label? Label { get; init; }
        public RectangleF Rectangle { get; init; }
        public float Score { get; init; }
    }
}
