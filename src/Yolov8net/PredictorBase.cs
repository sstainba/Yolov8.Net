using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using Yolov8Net.Extentions;

namespace Yolov8Net
{
    abstract public class PredictorBase
        : IPredictor
    {
        protected readonly InferenceSession _inferenceSession;
        protected string[] modelOutputs;
        protected Label[] Labels { get; set; } = new Label[] { };


        public float Confidence { get; protected set; } = 0.20f;
        public float MulConfidence { get; protected set; } = 0.25f;
        public float Overlap { get; protected set; } = 0.45f;
        public int ModelOutputDimensions { get; protected set; }

        public bool UseDetect { get; set; }


        protected PredictorBase(string modelPath, string[]? labels = null, bool useCuda = false)
        {

            if (useCuda)
                _inferenceSession = new InferenceSession(modelPath,
                    SessionOptions.MakeSessionOptionWithCudaProvider());
            else
                _inferenceSession = new InferenceSession(modelPath);

            /// Get model info
            GetInputDetails();
            GetOutputDetails();

            if (labels != null)
            {
                UseCustomLabels(labels);
            }
            else UseCoCoLabels();

            //if there are more labels than we have outputs for, throw an exception
            if (Labels?.Length + 4 > ModelOutputDimensions) throw new ArgumentOutOfRangeException("Number of labels provided exceeds output dimensions of model.");
        }

        public string? InputColumnName { get; protected set; }
        public string? OutputColumnName { get; protected set; }

        public int ModelInputHeight { get; protected set; }
        public int ModelInputWidth { get; protected set; }

        public void Dispose()
        {
            _inferenceSession?.Dispose();
        }

        protected void GetInputDetails()
        {
            InputColumnName = _inferenceSession.InputMetadata.Keys.First();
            ModelInputHeight = _inferenceSession.InputMetadata[InputColumnName].Dimensions[2];
            ModelInputWidth = _inferenceSession.InputMetadata[InputColumnName].Dimensions[3];
        }

        protected virtual void GetOutputDetails()
        {
            OutputColumnName = _inferenceSession.OutputMetadata.Keys.First();
            modelOutputs = _inferenceSession.OutputMetadata.Keys.ToArray();
            ModelOutputDimensions = _inferenceSession.OutputMetadata[modelOutputs[0]].Dimensions[1];
            UseDetect = !(modelOutputs.Any(x => x == "score"));
        }

        protected Prediction[] Suppress(List<Prediction> predictions)
        {
            var result = new List<Prediction>(predictions);

            foreach (var item in predictions) // iterate every prediction
            {
                foreach (var current in result.ToList()) // make a copy for each iteration
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    RectangleF intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Width * intersection.Height; // intersection area
                    float unionArea = rect1.Width * rect1.Height + rect2.Width * rect2.Height - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    if (overlap >= Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result.ToArray();
        }

        protected void UseCustomLabels(string[] labels)
        {
            Labels = labels.Select((s, i) => new { i, s }).ToList()
                .Select(x => new Label()
                {
                    Id = x.i,
                    Name = x.s
                }).ToArray();
        }

        protected void UseCoCoLabels()
        {
            var s = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
            UseCustomLabels(s);
        }

        protected virtual DenseTensor<float>[] Inference(Image img)
        {
            Bitmap resized = null;

            if (img.Width != ModelInputWidth || img.Height != ModelInputHeight)
            {
                resized = Utils.ResizeImage(img, ModelInputWidth, ModelInputHeight); // fit image size to specified input size
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

            foreach (var item in modelOutputs) // add outputs for processing
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
            };

            return output.ToArray();
        }

        public virtual Prediction[] Predict(Image image)
        {
            throw new NotImplementedException();
        }
    }
}
