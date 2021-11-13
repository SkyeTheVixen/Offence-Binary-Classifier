using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Discord.WebSocket;
using Discord;
using Discord.API;

namespace Offence_Binary_Classifier
{
    class Program
    {
        public class PhraseOffence //Training Data
        {
            [LoadColumn(0)] //as
            public bool Label { get; set; }
            [LoadColumn(1)] //as
            public string Phrase { get; set; }
        }

        public class OffencePrediction //Prediction Model
        {
            [ColumnName("PredictedLabel")] //as
            public bool Prediction { get; set; }
            public float Score { get; set; }
            public float Probability { get; set; }
        }

        private DiscordSocketClient _client;

        public async Task MainAsync()
        {
            _client = new DiscordSocketClient();
            _client.Log += Log;
            await _client.LoginAsync(TokenType.Bot, "TOKEN");
            await _client.StartAsync();
            await Task.Delay(-1);
        }

        private Task Log(LogMessage msg)
        {
            Console.WriteLine(msg.ToString());
            return Task.CompletedTask;
        }

        private Task checkMessage(string Message, PredictionEngine predictionEngine)
        {
            while (true)
            {
                Console.Write("Enter a phrase to classify: ");
                PhraseOffence phrase = new PhraseOffence { Phrase = Console.ReadLine() };

                var resultPrediction = predictionEngine.Predict(phrase);
                var predictionion = (Convert.ToBoolean(resultPrediction.Prediction) ? "Offensive" : "Not Offensive");
                Console.WriteLine("User Input: " + phrase.Phrase);
                Console.WriteLine("Prediction: " + predictionion);
                Console.WriteLine("Confidence: " + resultPrediction.Probability);
                Console.WriteLine("Score: " + resultPrediction.Score);

                Console.WriteLine("Would you like to classify another phrase? y/n");
                if (Console.ReadLine() == "n")
                {
                    break;
                }
            }
        }

        static void Main(string[] args)
        {
            PredictionEngine<PhraseOffence, OffencePrediction> predictionEngine;
            bool isModelLoaded = false;
            const string modelPath = "OffenceModel.zip";


            MLContext mLContext = new MLContext(); //Initialise new instance
            if (!File.Exists(modelPath) && isModelLoaded == false)
            {
                const string datasetPath = "trainingData.tsv";
                IDataView dataView = mLContext.Data.LoadFromTextFile<PhraseOffence>(datasetPath, hasHeader: true);
                TrainTestData trainTestSplit = mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
                IDataView trainingData = trainTestSplit.TrainSet;
                IDataView testData = trainTestSplit.TestSet;

                var dataPipeline = mLContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(PhraseOffence.Phrase));

                var trainer = mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
                var trainingPipeline = dataPipeline.Append(trainer);

                ITransformer trainedModel1 = trainingPipeline.Fit(trainingData);

                var predictions = trainedModel1.Transform(testData);
                var metrics = mLContext.BinaryClassification.Evaluate(predictions);

                Console.WriteLine("Accuracy: " + metrics.Accuracy);
                Console.WriteLine("Entropy: " + metrics.Entropy);
                Console.WriteLine("Area under Precission Recall Curve: " + metrics.AreaUnderPrecisionRecallCurve);
                Console.WriteLine(); //Output all stats


                Console.WriteLine("Would you like to save the results? y/n");
                if (Console.ReadLine() == "y")
                {
                    const string modelSavePath = "OffenceModel.zip";

                    mLContext.Model.Save(trainedModel1, trainingData.Schema, modelSavePath);
                    Console.WriteLine("Model Saved.");
                }

            }
            DataViewSchema predictionPipelineSchema;
            ITransformer trainedModel = mLContext.Model.Load(modelPath, out predictionPipelineSchema);
            predictionEngine = mLContext.Model.CreatePredictionEngine<PhraseOffence, OffencePrediction>(trainedModel);
            isModelLoaded = true;
            Console.WriteLine("Loaded model");
            new Program().MainAsync().GetAwaiter().GetResult();

            
        }
    }
}
