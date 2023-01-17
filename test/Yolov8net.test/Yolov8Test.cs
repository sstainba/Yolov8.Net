using System.Drawing;
using System.Drawing.Drawing2D;
using static System.Formats.Asn1.AsnWriter;

namespace Yolov8net.test
{
    public class Yolov8Test
    {
        [Fact]
        public void WowBobberTest()
        {
            using var yolo = new Yolov8("./assets/bobbers_v9_yolov8.onnx", false, new string[] { "bobber" });
            Assert.NotNull(yolo);

            using var image = Image.FromFile("Assets/bobber1.jpg");
            var predictions = yolo.Predict(image);

            Assert.NotNull(predictions);
            Assert.True(predictions.Count == 1);

            DrawBoxes(yolo.ModelInputHeight, yolo.ModelInputWidth, image, predictions);

            image.Save("bobber1-out.jpg");
        }

        [Fact]
        public void CocoTest()
        {
            using var yolo = new Yolov8("./assets/yolov8m.onnx");
            Assert.NotNull(yolo);

            using var image = Image.FromFile("Assets/input.jpg");
            var predictions = yolo.Predict(image);

            Assert.NotNull(predictions);

            DrawBoxes(yolo.ModelInputHeight, yolo.ModelInputWidth, image, predictions);

            image.Save("result.jpg");
        }

        [Fact]
        public void CocoTest_CUDA()
        {
            using var yolo = new Yolov8("./assets/yolov8m.onnx", true);
            Assert.NotNull(yolo);

            using var image = Image.FromFile("Assets/input.jpg");
            var predictions = yolo.Predict(image);

            Assert.NotNull(predictions);

            DrawBoxes(yolo.ModelInputHeight, yolo.ModelInputWidth, image, predictions);

            image.Save("result.jpg");
        }

        private void DrawBoxes(int modelInputHeight, int modelInputWidth, Image image, List<Prediction> predictions)
        {
            foreach (var pred in predictions)
            {
                var originalImageHeight = image.Height;
                var originalImageWidth = image.Width;

                var x = Math.Max(pred.Rectangle.X, 0);
                var y = Math.Max(pred.Rectangle.Y, 0);
                var width = Math.Min(originalImageWidth - x, pred.Rectangle.Width);
                var height = Math.Min(originalImageHeight - y, pred.Rectangle.Height);

                //Note that the output is already scaled to the original image height and width.

                // Bounding Box Text
                string text = $"{pred.Label.Name} [{pred.Score}]";

                using (Graphics graphics = Graphics.FromImage(image))
                {
                    graphics.CompositingQuality = CompositingQuality.HighQuality;
                    graphics.SmoothingMode = SmoothingMode.HighQuality;
                    graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    // Define Text Options
                    Font drawFont = new Font("consolas", 11, FontStyle.Regular);
                    SizeF size = graphics.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                    // Define BoundingBox options
                    Pen pen = new Pen(Color.Yellow, 2.0f);
                    SolidBrush colorBrush = new SolidBrush(Color.Yellow);

                    // Draw text on image 
                    graphics.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    graphics.DrawString(text, drawFont, fontBrush, atPoint);

                    // Draw bounding box on image
                    graphics.DrawRectangle(pen, x, y, width, height);
                }
            }
        }
    }
}
