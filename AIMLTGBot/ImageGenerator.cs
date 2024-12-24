using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace AIMLTGBot
{
    /// <summary>
    /// Тип фигуры
    /// </summary>
    public enum FigureType : byte { Angry = 0, Happy, Neutral, Sad, Surprised, Undef };
    
    public class GenerateImage
    {
        /// <summary>
        /// Текущая сгенерированная фигура
        /// </summary>
        public FigureType currentFigure = FigureType.Undef;

        /// <summary>
        /// Количество классов генерируемых фигур (5 - максимум)
        /// </summary>
        public int EmotionsCount { get; set; } = 5;


        public List<Sample> LoadTrainSamples()
        {
            string path = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName + "\\dataset\\train";

            var smiles = Directory.GetFiles(path + "\\happy").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Happy));
            var sads = Directory.GetFiles(path + "\\sad").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Sad));
            var angries = Directory.GetFiles(path + "\\angry").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Angry));
            var neutrals = Directory.GetFiles(path + "\\neutral").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Neutral));
            var surpriseds = Directory.GetFiles(path + "\\surprised").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Surprised));

            return smiles
                .Concat(sads)
                .Concat(angries)
                .Concat(neutrals)
                .Concat(surpriseds)
                .ToList();
        }

        public List<Sample> LoadTestSamples()
        {
            string path = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName + "\\dataset\\test";

            var smiles = Directory.GetFiles(path + "\\happy").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Happy));
            var sads = Directory.GetFiles(path + "\\sad").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Sad));
            var angries = Directory.GetFiles(path + "\\angry").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Angry));
            var neutrals = Directory.GetFiles(path + "\\neutral").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Neutral));
            var surpriseds = Directory.GetFiles(path + "\\surprised").Select(filename => new Sample(ImageEncoder.Flatten(new Bitmap(filename)), EmotionsCount, FigureType.Surprised));

            return smiles
                .Concat(sads)
                .Concat(angries)
                .Concat(neutrals)
                .Concat(surpriseds)
                .ToList();
        }
    }
}
