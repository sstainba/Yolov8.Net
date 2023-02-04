using System.Drawing;
using System.Drawing.Drawing2D;

namespace Yolov8Net.test
{
    public class Yolov8Test
    {
        [Fact]
        public void WowBobberTest()
        {
            string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "output_WowBobberTest");
            if (Directory.Exists(outputPath))
            {
                Directory.Delete(outputPath, true);
            }
            Directory.CreateDirectory(outputPath);


            using var yolo = YoloV8Predictor.Create("./assets/bobbers_v9_yolov8.onnx", new string[] { "bobber" });
            Assert.NotNull(yolo);

            var inputFiles = Directory.GetFiles("./assets/", "bob*.jpg");

            foreach (var inputFile in inputFiles)
            {
                var fileName = Path.GetFileNameWithoutExtension(inputFile);
                using var image = Image.FromFile(inputFile);
                var predictions = yolo.Predict(image);

                Assert.NotNull(predictions);

                DrawBoxes(yolo.ModelInputHeight, yolo.ModelInputWidth, image, predictions);

                image.Save(Path.Combine(outputPath, $"{fileName}.jpg"));
            }
        }

        [Fact]
        public void WowBobberLableMismatchTest()
        {

            string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "output_WowBobberLableMismatchTest");
            if (Directory.Exists(outputPath))
            {
                Directory.Delete(outputPath, true);
            }
            Directory.CreateDirectory(outputPath);

            IPredictor yolo = null;

            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                yolo = YoloV8Predictor.Create("./assets/bobbers_v9_yolov8.onnx");
            });
        }

        [Fact]
        public void CocoTest()
        {
            string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "output_CocoTest");
            if (Directory.Exists(outputPath))
            {
                Directory.Delete(outputPath, true);
            }
            Directory.CreateDirectory(outputPath);

            using var yolo = YoloV8Predictor.Create("./assets/yolov8m.onnx");
            Assert.NotNull(yolo);

            using var image = Image.FromFile("Assets/input.jpg");
            var predictions = yolo.Predict(image);

            Assert.NotNull(predictions);

            DrawBoxes(yolo.ModelInputHeight, yolo.ModelInputWidth, image, predictions);

            image.Save(Path.Combine(outputPath, "result.jpg"));
        }

        [Fact]
        public void CocoTest_CUDA()
        {
            // NOTE:  Must have CUDA Dev pack installed


            using var yolo = YoloV8Predictor.Create("./assets/yolov8m.onnx", null, true);
            Assert.NotNull(yolo);

            using var image = Image.FromFile("Assets/input.jpg");
            var predictions = yolo.Predict(image);

            Assert.NotNull(predictions);

            DrawBoxes(yolo.ModelInputHeight, yolo.ModelInputWidth, image, predictions);

            image.Save("result.jpg");
        }

        private void DrawBoxes(int modelInputHeight, int modelInputWidth, Image image, Prediction[] predictions)
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
