using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace NeuralNetwork1
{
    public static class ImageEncoder
    {
        private static int sampleX = 30;
        private static int sampleY = 30;
        /// <summary>
        /// Преобразует картинку в массив длиной 900
        /// </summary>
        /// <param name="original">Исходная картинка</param>
        /// <returns>Массив интенсивностей</returns>
        public static double[] Flatten(Bitmap original)
        {
            var blob = ExtractBlob(original);
            var vector = Vectorize(blob);
            return vector;
        }

        private static Bitmap ExtractBlob(Bitmap original)
        {
            // Переводим в оттенки серого
            var grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
            var unmanaged = grayFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(original));

            // Применяем пороговый фильтр
            var thresholdFilter = new AForge.Imaging.Filters.OtsuThreshold();
            thresholdFilter.ApplyInPlace(unmanaged);

            // Инвертируем
            var invertFilter = new AForge.Imaging.Filters.Invert();
            invertFilter.ApplyInPlace(unmanaged);

            // Создаём BlobCounter, вытаскиваем блобы
            var bc = new AForge.Imaging.BlobCounter();

            bc.FilterBlobs = true;
            bc.MinWidth = 5;
            bc.MinHeight = 5;
            // Упорядочиваем по размеру
            bc.ObjectsOrder = AForge.Imaging.ObjectsOrder.Size;

            bc.ProcessImage(unmanaged);

            var rectangles = bc.GetObjectsRectangles();

            // Находим выпуклую оболочку
            int lx = unmanaged.Width;
            int ly = unmanaged.Height;
            int rx = 0;
            int ry = 0;
            for (int i = 0; i < rectangles.Length; ++i)
            {
                if (lx > rectangles[i].X) lx = rectangles[i].X;
                if (ly > rectangles[i].Y) ly = rectangles[i].Y;
                if (rx < rectangles[i].X + rectangles[i].Width) rx = rectangles[i].X + rectangles[i].Width;
                if (ry < rectangles[i].Y + rectangles[i].Height) ry = rectangles[i].Y + rectangles[i].Height;
            }

            // Обрезаем края, оставляя только центральные блобчики
            var cropFilter = new AForge.Imaging.Filters.Crop(new Rectangle(lx, ly, rx - lx, ry - ly));
            unmanaged = cropFilter.Apply(unmanaged);

            //  Масштабируем до 30x30
            var scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(sampleX, sampleY);
            unmanaged = scaleFilter.Apply(unmanaged);

            return unmanaged.ToManagedImage();
        }

        private static double[] Vectorize(Bitmap blob)
        {
            var vector = Enumerable
                .Repeat(0.0, blob.Width * blob.Height)
                .ToArray();
            int i = 0;
            for (int x = 0; x < blob.Width; x += 1)
            {
                for (int y = 0; y < blob.Height; y += 1)
                {
                    var color = blob.GetPixel(x, y);
                    vector[i++] = color.R / 255.0;
                }
            }
            return vector;
        }
    }
}
