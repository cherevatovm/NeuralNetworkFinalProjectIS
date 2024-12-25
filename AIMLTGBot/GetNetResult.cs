using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace AIMLTGBot
{
    class GetNetResult
    {
        private StudentNetwork network;

        public GetNetResult(StudentNetwork network)
        {
            this.network = network;
        }

        public string Result(Bitmap img) {
            double[] S = ImageEncoder.Flatten(img);
            Sample sample = new Sample(S, 5);
            sample.ProcessPrediction(network.Compute(sample.input));

            switch (sample.recognizedClass)
            {
                case FigureType.Angry:
                    return "angry";
                case FigureType.Happy:
                    return "happy";
                case FigureType.Neutral:
                    return "neutral";
                case FigureType.Sad:
                    return "sad";
                default:
                    return "surprised";
            }
        }
    }
}