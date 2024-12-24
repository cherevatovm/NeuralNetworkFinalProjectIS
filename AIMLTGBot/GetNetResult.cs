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

        public string Result(Bitmap img)
        {
            double[] S = ImageEncoder.Flatten(img);
            double[] res = network.Compute(S);

            int j = 0;
            double max = 0;
            for (int i = 0; i < res.Length; i++)
            {
                if (res[i] > max)
                {
                    max = res[i];
                    j = i;
                }
            }

            string str = "";
            switch (j)
            {
                case 0:
                    str = "angry";
                    break;
                case 1:
                    str = "happy";
                    break;
                case 2:
                    str = "neutral";
                    break;
                case 3:
                    str = "sad";
                    break;
                case 4:
                    str = "surprised";
                    break;
                default:
                    break;
            }
            return str;
        }
    }
}