using System.Drawing;

namespace Yolov8net
{
    public class Label
    {
        public int Id { get; set; }
        public string? Name { get; set; }
        public LabelKind Kind { get; set; }
    }
}
