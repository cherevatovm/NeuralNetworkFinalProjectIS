using Accord;
using Accord.MachineLearning.Boosting;
using Accord.Statistics.Kernels;
using System;
using System.CodeDom;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using Accord.Neuro;
using System.IO;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        Stopwatch watch = new Stopwatch();
        static Random rand = new Random();
        int[] savedStruct;

        public static double learningRate = 0.1;

        List<Neuron[]> allLayers;

        public StudentNetwork(int[] structure)
        {
            savedStruct = structure;
            Init(structure);
        }

        private void Init(int[] structure)
        {
            if (structure.Length < 2)
            {
                throw new ArgumentException("invalid structure network");
            }
            allLayers = new List<Neuron[]>
            {
                new Neuron[structure[0]]
            };
            for (int i = 0; i < structure[0]; ++i)
                allLayers[0][i] = new Neuron();
            for (int i = 1; i < structure.Length; i++)
            {
                allLayers.Add(new Neuron[structure[i]]);
                for (int j = 0; j < structure[i]; j++)
                    allLayers[i][j] = new Neuron(allLayers[i - 1]);
            }

            foreach (var layer in allLayers)
            {
                foreach (Neuron neuron in layer)
                {
                    if (neuron.weights == null)
                        continue;
                    double stdDev = 1.0 / Math.Sqrt(neuron.weights.Length);
                    for (int i = 0; i < neuron.weights.Length; i++)
                        neuron.weights[i] = rand.NextDouble() * 2 * stdDev - stdDev;
                    neuron.bias = 2 * rand.NextDouble() * stdDev - stdDev;
                }
            }
        }

        protected override double[] Compute(double[] input)
        {
            Neuron[] inputLayer = allLayers.First();
            for (int i = 0; i < inputLayer.Length; ++i)
                inputLayer[i].output = input[i];
            for (int i = 1; i < allLayers.Count; i++)
                for (int j = 0; j < allLayers[i].Length; j++)
                    allLayers[i][j].ForwardPropagation();
            return allLayers.Last().Select(x => x.output).ToArray();
        }

        private void BackPropagation(Sample samp)
        {
            Neuron[] outputLayer = allLayers.Last();
            for (int i = 0; i < outputLayer.Length; ++i)
                outputLayer[i].error = samp.error[i];
            for (int i = allLayers.Count - 2; i > 0; --i)
            {
                for (int j = 0; j < allLayers[i].Length; ++j)
                {
                    double scalar = 0;
                    for (int k = 0; k < allLayers[i + 1].Length; ++k)
                        scalar += allLayers[i + 1][k].error * allLayers[i + 1][k].weights[j];
                    allLayers[i][j].error = scalar * Neuron.actFuncDeriv(allLayers[i][j].output);
                }
            }
            for (int i = 0; i < outputLayer.Length; ++i)
                outputLayer[i].WeightAdjustment(learningRate);
            for (int i = allLayers.Count - 2; i > 0; --i)
            {
                for (int j = 0; j < allLayers[i].Length; ++j)
                    allLayers[i][j].WeightAdjustment(learningRate);
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel = false)
        {
            var iter = 0;
            sample.ProcessPrediction(Compute(sample.input));
            var error = sample.EstimatedError();
            while (error > acceptableError)
            {
                iter++;
                sample.ProcessPrediction(Compute(sample.input));
                error = sample.EstimatedError();
                BackPropagation(sample);
            }
            return iter;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel = false)
        {
            watch.Restart();
            double error = 0;
            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                double errorSum = 0;
                foreach (var sample in samplesSet.samples)
                {
                    int TrainResult = Train(sample, acceptableError);
                    if (TrainResult == 0)
                        errorSum += sample.EstimatedError();
                }
                error = errorSum;
                OnTrainProgress(((epoch + 1) * 1.0) / epochsCount, error, watch.Elapsed);
            }
            watch.Stop();
            return error;
        }

        private string path = "../../saved-network.txt";

        public override void Save()
        {
            using (StreamWriter writer = new StreamWriter(path))
            {
                writer.WriteLine(string.Join(",", savedStruct.Select(i => i.ToString())));
                foreach (var layer in allLayers.Skip(1))
                {
                    foreach (var neuron in layer)
                    {
                        writer.WriteLine(neuron.Serialize());
                    }
                }
            }
        }

        public override void Load()
        {
            var lines = File.ReadAllLines(path);
            int[] structure = lines[0].Split(',').Select(int.Parse).ToArray();
            int lineIndex = 1;
            Init(structure);
            foreach (var layer in allLayers.Skip(1))
            {
                foreach (var neuron in layer)
                {
                    var data = lines[lineIndex].Split(';');
                    var weightStrings = data[0].Split(':');

                    for (int i = 0; i < weightStrings.Length; i++)
                    {
                        neuron.weights[i] = double.Parse(weightStrings[i]);
                    }
                    neuron.bias = double.Parse(data[1]);
                    lineIndex++;
                }
            }
        }
    }
}