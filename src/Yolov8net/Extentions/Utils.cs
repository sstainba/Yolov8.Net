using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.Advanced;
using System.Diagnostics;

namespace Yolov8Net.Extentions
{
    public static class Utils
    {
        public static float[] Xywh2xyxy(float[] source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

        public static Image ResizeImage(Image image,int target_width,int target_height)
        {
            return image.Clone(x => x.Resize(target_width,target_height));
        }

        public static Tensor<float> ExtractPixels(Image image)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });

            using (var img = image.CloneAs<Rgb24>())
            {
                Parallel.For(0, img.Height, y => {
                    var pixelSpan = img.DangerousGetPixelRowMemory((int)y).Span;
                    for(int x = 0; x < img.Width;x++)
                    {
                        tensor[0, 0, y, x] = pixelSpan[x].R / 255.0F; // r
                        tensor[0, 1, y, x] = pixelSpan[x].G / 255.0F; // g
                        tensor[0, 2, y, x] = pixelSpan[x].B / 255.0F; // b
                    }
                });
            }
            return tensor;
        }

        public static float Clamp(float value, float min, float max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }

        public static float Sigmoid(float value)
        {
            return 1 / (1 + (float)Math.Exp(-value));
        }
    }
}
