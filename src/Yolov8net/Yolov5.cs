using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using Yolov8Net.Extentions;

namespace Yolov8Net
{
    public class YoloV5Predictor
        : PredictorBase, IPredictor
    {
        /// <summary>
        /// Create a YoloV5 Predictor.
        /// </summary>
        /// <param name="modelPath">Path to the ONNX format model to load.</param>
        /// <param name="labels">Labels associated with model. If not provided, standard COCO labels are used.</param>
        /// <param name="useCuda">Use GPU/CUDA.  NOTE: Requires CUDA drivers AND CUDNN be installed.</param>
        /// <returns>IPredictor</returns>
        public static IPredictor Create(string modelPath, string[]? labels = null, bool useCuda = false)
        {
            return new YoloV5Predictor(modelPath, labels, useCuda);
        }

        private YoloV5Predictor(string modelPath, string[]? labels = null, bool useCuda = false)
            : base(modelPath, labels, useCuda) { }

        public int[] Strides { get; set; } = new int[] { 8, 16, 32 };

        public int[][][] Anchors { get; set; } = new int[][][]
        {
            new int[][] { new int[] { 010, 13 }, new int[] { 016, 030 }, new int[] { 033, 023 } },
            new int[][] { new int[] { 030, 61 }, new int[] { 062, 045 }, new int[] { 059, 119 } },
            new int[][] { new int[] { 116, 90 }, new int[] { 156, 198 }, new int[] { 373, 326 } }
        };

        public int[] Shapes { get; set; } = new int[] { 80, 40, 20 };

        override protected void GetOutputDetails()
        {
            OutputColumnName = _inferenceSession.OutputMetadata.Keys.First();
            modelOutputs = _inferenceSession.OutputMetadata.Keys.ToArray();
            ModelOutputDimensions = _inferenceSession.OutputMetadata[modelOutputs[0]].Dimensions[2];
            UseDetect = !(modelOutputs.Any(x => x == "score"));
        }

        private List<Prediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var result = new ConcurrentBag<Prediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (ModelInputWidth / (float)w, ModelInputHeight / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((ModelInputWidth - w * gain) / 2, (ModelInputHeight - h * gain) / 2); // left, right pads

            Parallel.For(0, (int)output.Length / ModelOutputDimensions, (i) =>
            {
                if (output[0, i, 4] <= Confidence) return; // skip low obj_conf results

                Parallel.For(5, ModelOutputDimensions, (j) =>
                {
                    output[0, i, j] = output[0, i, j] * output[0, i, 4]; // mul_conf = obj_conf * cls_conf
                });

                Parallel.For(5, ModelOutputDimensions, (k) =>
                {
                    if (output[0, i, k] <= MulConfidence) return; // skip low mul_conf results

                    float xMin = ((output[0, i, 0] - output[0, i, 2] / 2) - xPad) / gain; // unpad bbox tlx to original
                    float yMin = ((output[0, i, 1] - output[0, i, 3] / 2) - yPad) / gain; // unpad bbox tly to original
                    float xMax = ((output[0, i, 0] + output[0, i, 2] / 2) - xPad) / gain; // unpad bbox brx to original
                    float yMax = ((output[0, i, 1] + output[0, i, 3] / 2) - yPad) / gain; // unpad bbox bry to original

                    xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                    yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                    xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                    yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                    Label label = Labels[k - 5];

                    var prediction = new Prediction()
                    {
                        Label = label,
                        Score = output[0, i, k],
                        Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                    };

                    result.Add(prediction);
                });
            });

            return result.ToList();
        }

        private List<Prediction> ParseSigmoid(DenseTensor<float>[] output, Image image)
        {
            var result = new ConcurrentBag<Prediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (ModelInputWidth / (float)w, ModelInputHeight / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((ModelInputWidth - w * gain) / 2, (ModelInputHeight - h * gain) / 2); // left, right pads

            Parallel.For(0, output.Length, (i) => // iterate model outputs
            {
                int shapes = Shapes[i]; // shapes per output

                Parallel.For(0, Anchors[0].Length, (a) => // iterate anchors
                {
                    Parallel.For(0, shapes, (y) => // iterate shapes (rows)
                    {
                        Parallel.For(0, shapes, (x) => // iterate shapes (columns)
                        {
                            int offset = (shapes * shapes * a + shapes * y + x) * ModelOutputDimensions;

                            float[] buffer = output[i].Skip(offset).Take(ModelOutputDimensions).Select(Utils.Sigmoid).ToArray();

                            if (buffer[4] <= Confidence) return; // skip low obj_conf results

                            List<float> scores = buffer.Skip(5).Select(b => b * buffer[4]).ToList(); // mul_conf = obj_conf * cls_conf

                            float mulConfidence = scores.Max(); // max confidence score

                            if (mulConfidence <= MulConfidence) return; // skip low mul_conf results

                            float rawX = (buffer[0] * 2 - 0.5f + x) * Strides[i]; // predicted bbox x (center)
                            float rawY = (buffer[1] * 2 - 0.5f + y) * Strides[i]; // predicted bbox y (center)

                            float rawW = (float)Math.Pow(buffer[2] * 2, 2) * Anchors[i][a][0]; // predicted bbox w
                            float rawH = (float)Math.Pow(buffer[3] * 2, 2) * Anchors[i][a][1]; // predicted bbox h

                            float[] xyxy = Utils.Xywh2xyxy(new float[] { rawX, rawY, rawW, rawH });

                            float xMin = Utils.Clamp((xyxy[0] - xPad) / gain, 0, w - 0); // unpad, clip tlx
                            float yMin = Utils.Clamp((xyxy[1] - yPad) / gain, 0, h - 0); // unpad, clip tly
                            float xMax = Utils.Clamp((xyxy[2] - xPad) / gain, 0, w - 1); // unpad, clip brx
                            float yMax = Utils.Clamp((xyxy[3] - yPad) / gain, 0, h - 1); // unpad, clip bry

                            Label label = Labels[scores.IndexOf(mulConfidence)];

                            var prediction = new Prediction()
                            {
                                Label = label,
                                Score = mulConfidence,
                                Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                            };

                            result.Add(prediction);
                        });
                    });
                });
            });

            return result.ToList();
        }

        protected List<Prediction> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            return UseDetect ? ParseDetect(output[0], image) : ParseSigmoid(output, image);

        }

        public override Prediction[] Predict(Image image)
        {
            return Suppress(
             ParseOutput(
                 Inference(image), image)
             );
        }

    }
}
