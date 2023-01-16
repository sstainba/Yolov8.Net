
using System.Diagnostics;
using System.Drawing;


namespace Yolov8net.test
{
    public class Yolov8Test
    {
        [Fact]
        public void TestYolov8()
        {
            
            using var yolo = new Yolov8("./assets/bobbers_v9_yolov8_opset_12.onnx", false, new string[] { "bobber" });
            Assert.NotNull(yolo);
            
            using var image = Image.FromFile("Assets/bobber1.jpg");
            var ret = yolo.Predict(image);

            Assert.NotNull(ret);
            Assert.True(ret.Count == 1);
        }
    }
}
