using Accord.Neuro;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class Neuron
    {
        public static Func<double, double> actFunc = x => 1.0 / (1.0 + Math.Exp(-x));
        public static Func<double, double> actFuncDeriv = x => x * (1 - x);

        public Neuron[] prevLayerWeights;
        //выходной сигнал
        public double output;
        //веса
        public double[] weights;
        //значение ошибки
        public double error;
        // биас
        public double bias;

        public Neuron() {}
        public Neuron(Neuron[] prevLayer)
        {
            prevLayerWeights = prevLayer;
            weights = new double[prevLayer.Length];
        }
        /// <summary>
        /// Запускает работу нейрона. Присваивает выходу значение функции активации
        /// </summary>
        public void ForwardPropagation()
        {
            double res = bias;
            for (int i = 0; i < prevLayerWeights.Length; ++i)
                res += prevLayerWeights[i].output * weights[i]; // взвешенная сумма сигналов
            output = actFunc(res);
        }

        public void WeightAdjustment(double learningRate)
        {
            for (int i = 0; i < weights.Length; ++i)
                weights[i] -= learningRate * error * prevLayerWeights[i].output;
            bias -= learningRate * error;
        }
    }
}
