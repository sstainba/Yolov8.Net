using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing;
using Yolov8net.Extentions;
using Yolov8net.Models;

namespace Yolov8net
{
    public class Yolov8
        : IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private YoloModel _model = new YoloModel();

        public Yolov8(string ModelPath, bool useCuda = false, string[]? labels = null)
        {

            if (useCuda)
            {
                SessionOptions opts = SessionOptions.MakeSessionOptionWithCudaProvider();
                _inferenceSession = new InferenceSession(ModelPath, opts);
            }
            else
            {
                SessionOptions opts = new();
                _inferenceSession = new InferenceSession(ModelPath, opts);
            }


            /// Get model info
            get_input_details();
            get_output_details();

            if (labels != null)
            {
                UseCustomLabels(labels);
            }
            else UseDefaultLabels();
        }

        public string? InputColumnName { get; private set; }
        public string? OutputColumnName { get; private set; }

        public int ModelInputHeight => _model?.Height ?? 0;
        public int ModelInputWidth => _model?.Width ?? 0;
        public int ModelInputDepth => _model?.Depth ?? 0;

        public List<Prediction> Predict(Image image)
        {
            return Suppress(ParseDetect(Inference(image)[0], image));
        }

        private void UseCustomLabels(string[] labels)
        {
            labels.Select((s, i) => new { i, s }).ToList().ForEach(item =>
            {
                _model.Labels.Add(new Label { Id = item.i, Name = item.s });
            });
        }

        private void UseDefaultLabels()
        {
            var s = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
            UseCustomLabels(s);
        }

        private List<Prediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var result = new ConcurrentBag<Prediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // left, right pads

            //for each batch
            Parallel.For(0, output.Dimensions[0], i =>
            {
                //divide total length by the elements per prediction
                Parallel.For(0, (int)(output.Length / output.Dimensions[1]), j => {

                    float xMin = ((output[i, 0, j] - output[i, 2, j] / 2) - xPad) / gain; // unpad bbox tlx to original
                    float yMin = ((output[i, 1, j] - output[i, 3, j] / 2) - yPad) / gain; // unpad bbox tly to original
                    float xMax = ((output[i, 0, j] + output[i, 2, j] / 2) - xPad) / gain; // unpad bbox brx to original
                    float yMax = ((output[i, 1, j] + output[i, 3, j] / 2) - yPad) / gain; // unpad bbox bry to original

                    xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                    yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                    xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                    yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                    Parallel.For(0, _model.Labels.Count, l =>
                    {
                        var pred = output[i, 4 + l, j];

                        //skip low confidence values
                        if (pred < _model.Confidence) return;

                        result.Add(new Prediction(_model.Labels[l], pred)
                        {
                            Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                        });
                    });

                });
                
            });

            return result.ToList();
        }

        private DenseTensor<float>[] Inference(Image img)
        {
            Bitmap resized = null;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height); // fit image size to specified input size
            }
            else
            {
                resized = new Bitmap(img);
            }

            var inputs = new List<NamedOnnxValue> // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor(InputColumnName, Utils.ExtractPixels(resized))
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _inferenceSession.Run(inputs); // run inference

            var output = new List<DenseTensor<float>>();

            foreach (var item in _model.Outputs) // add outputs for processing
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
            };

            return output.ToArray();
        }

        private List<Prediction> Suppress(List<Prediction> items)
        {
            var result = new List<Prediction>(items);

            foreach (var item in items) // iterate every prediction
            {
                foreach (var current in result.ToList()) // make a copy for each iteration
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    RectangleF intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Width * intersection.Height; // intersection area
                    float unionArea = rect1.Width * rect1.Height + rect2.Width * rect2.Height - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    if (overlap >= _model.Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result;
        }

        private void get_input_details()
        {
            InputColumnName = _inferenceSession.InputMetadata.Keys.First();
            _model.Height = _inferenceSession.InputMetadata[InputColumnName].Dimensions[2];
            _model.Width = _inferenceSession.InputMetadata[InputColumnName].Dimensions[3];
        }

        private void get_output_details()
        {
            OutputColumnName = _inferenceSession.OutputMetadata.Keys.First();
            _model.Outputs = _inferenceSession.OutputMetadata.Keys.ToArray();
            _model.Dimensions = _inferenceSession.OutputMetadata[_model.Outputs[0]].Dimensions[1];
            _model.UseDetect = !(_model.Outputs.Any(x => x == "score"));
        }

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}
