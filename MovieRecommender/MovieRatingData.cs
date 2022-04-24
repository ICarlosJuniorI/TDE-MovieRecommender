using Microsoft.ML.Data;

namespace MovieRecommender
{
    //Especifica uma classe de dados de entrada
    public class MovieRating
    {
        //LoadColumn especifica quais colunas devem ser carregadas
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float movieId;
        [LoadColumn(2)]
        public float Label;
    }
}
