using Microsoft.ML;
using Microsoft.ML.Trainers;
using MovieRecommender;

MLContext mlContext = new MLContext();

var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

//Carrega os dados
(IDataView training, IDataView test) LoadData(MLContext mlContext)
{
    return (trainingDataView, testDataView);
}

ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

//Realiza o treinamento do modelo
ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
{
    IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

    //MatrixFactorizationTrainer é o algoritmo de treinamento de recomendação
    var options = new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = "userIdEncoded",
        MatrixRowIndexColumnName = "movieIdEncoded",
        LabelColumnName = "Label",
        NumberOfIterations = 20,
        ApproximationRank = 100
    };

    var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

    Console.WriteLine("=============== Training the model ===============");
    ITransformer model = trainerEstimator.Fit(trainingDataView);

    return model;
}

EvaluateModel(mlContext, testDataView, model);
UseModelForSinglePrediction(mlContext, model);
SaveModel(mlContext, trainingDataView.Schema, model);

//Usa os dados de teste para avaliar o desempenho
void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
{
    Console.WriteLine("=============== Evaluating the model ===============");
    var prediction = model.Transform(testDataView);

    var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

    Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
    Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
}

//Prevendo novos dados
void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
{
    //Preve a classificação
    Console.WriteLine("=============== Making a prediction ===============");
    var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

    var testInput = new MovieRating { userId = 6, movieId = 10 };

    //Predict() faz uma previsão sobre uma única coluna de dados
    var movieRatingPrediction = predictionEngine.Predict(testInput);

    //Se a classificação for maior que 3.5 então o filme é recomendado para o usuário
    if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
    {
        Console.WriteLine("Movie " + testInput.movieId + " is recommended for user " + testInput.userId);
    } else
    {
        Console.WriteLine("Movie " + testInput.movieId + " is not recommended for user " + testInput.userId);
    }
}

//Salva o modelo treinado em um arquivo .zip na pasta Data
void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");

    Console.WriteLine("=============== Saving the model to a file ===============");
    mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
}