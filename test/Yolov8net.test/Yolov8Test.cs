using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;

namespace Yolov8Net.test
{
    public class Yolov8Test
    {

        private readonly Font font;

        public Yolov8Test()
        {
            var fontCollection = new FontCollection();
            var fontFamily = fontCollection.Add("assets/CONSOLA.TTF");
            font = fontFamily.CreateFont(11, FontStyle.Bold);
        }

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
                using var image = Image.Load(inputFile);
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

            using var image = Image.Load("Assets/input.jpg");
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

            using var image = Image.Load("Assets/input.jpg");
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

                var x = (int)Math.Max(pred.Rectangle.X, 0);
                var y = (int)Math.Max(pred.Rectangle.Y, 0);
                var width = (int)Math.Min(originalImageWidth - x, pred.Rectangle.Width);
                var height = (int)Math.Min(originalImageHeight - y, pred.Rectangle.Height);

                //Note that the output is already scaled to the original image height and width.

                // Bounding Box Text
                string text = $"{pred.Label.Name} [{pred.Score}]";
                var size = TextMeasurer.MeasureSize(text, new TextOptions(font));

                image.Mutate(d => d.Draw(Pens.Solid(Color.Yellow, 2),
                    new Rectangle(x, y, width, height)));

                image.Mutate(d => d.DrawText(text, font, Color.Yellow, new Point(x, (int)(y - size.Height - 1))));
                    
            }
        }
    }
}
