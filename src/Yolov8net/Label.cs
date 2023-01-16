using System.Drawing;

namespace Yolov8net
{
    public class Label
    {
        public int Id { get; set; }
        public string? Name { get; set; }
        public LabelKind Kind { get; set; }
        public Color Color { get; set; }

        public Label() => Color = Color.Yellow;
    }
}
