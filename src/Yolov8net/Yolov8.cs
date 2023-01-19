using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing;
using Yolonet.Extentions;

namespace Yolonet
{
    public class YoloV8Predictor
        : PredictorBase, IPredictor
    {
        public static IPredictor Create(string modelPath, string[]? labels = null, bool useCuda = false)
        {
            return new YoloV8Predictor(modelPath, labels, useCuda);
        }

        private YoloV8Predictor(string modelPath, string[]? labels = null, bool useCuda = false)
            : base(modelPath, labels, useCuda) { }

        protected override List<Prediction> ParseOutput(DenseTensor<float> output, Image image)
        {
            var result = new ConcurrentBag<Prediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (ModelInputWidth / (float)w, ModelInputHeight / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((ModelInputWidth - w * gain) / 2, (ModelInputHeight - h * gain) / 2); // left, right pads

            //for each batch
            Parallel.For(0, output.Dimensions[0], i =>
            {
                //divide total length by the elements per prediction
                Parallel.For(0, (int)(output.Length / output.Dimensions[1]), j =>
                {

                    float xMin = ((output[i, 0, j] - output[i, 2, j] / 2) - xPad) / gain; // unpad bbox tlx to original
                    float yMin = ((output[i, 1, j] - output[i, 3, j] / 2) - yPad) / gain; // unpad bbox tly to original
                    float xMax = ((output[i, 0, j] + output[i, 2, j] / 2) - xPad) / gain; // unpad bbox brx to original
                    float yMax = ((output[i, 1, j] + output[i, 3, j] / 2) - yPad) / gain; // unpad bbox bry to original

                    xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                    yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                    xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                    yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                    Parallel.For(0, ModelOutputDimensions - 4, l =>
                    {
                        var pred = output[i, 4 + l, j];

                        //skip low confidence values
                        if (pred < Confidence) return;

                        result.Add(new Prediction()
                        {
                            Label = Labels[l],
                            Score = pred,
                            Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                        });
                    });
                });
            });

            return result.ToList();
        }
    }
}
