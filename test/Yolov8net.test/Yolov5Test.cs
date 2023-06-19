using SixLabors.Fonts;
using SixLabors.ImageSharp.Drawing.Processing;


namespace Yolov8Net.test
{
    public class Yolov5Test
    {
        private readonly Font font;

        public Yolov5Test()
        {
            var fontCollection = new FontCollection();
            var fontFamily = fontCollection.Add("assets/CONSOLA.TTF");
            font = fontFamily.CreateFont(11);
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


            using var yolo = YoloV5Predictor.Create("./assets/bobbers_v5_m.onnx", new string[] { "bobber" });
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
                yolo = YoloV5Predictor.Create("./assets/bobbers_v5_m.onnx");
            });
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
                var size = TextMeasurer.Measure(text, new TextOptions(font));

                image.Mutate(d => d.Draw(Pens.Solid(Color.Yellow, 2),
                    new Rectangle(x, y, width, height)));


                image.Mutate(d => d.DrawText(
                    new TextOptions(font)
                    {
                        Origin = new Point(x, (int)(y - size.Height - 1))
                    },
                    text, Color.Yellow)); ;
            }
        }
    }
}
