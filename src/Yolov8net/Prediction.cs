using System.Drawing;

namespace Yolov8net
{
    public class Prediction
    {
        public Label? Label { get; set; }
        public RectangleF Rectangle { get; set; }
        public float Score { get; set; }

        public Prediction() { }

        public Prediction(Label label, float score)
            : this(label)
        {
            Score = score;
        }

        public Prediction(Label label)
        {
            Label = label;
        }
    }
}
