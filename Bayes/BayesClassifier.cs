using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Text.RegularExpressions;

// This class is largely a port of https://github.com/jekyll/classifier-reborn/blob/master/lib/classifier-reborn/bayes.rb

[Serializable]
public sealed class BayesClassifier : ISerializable
{
    private static readonly Regex WordRegex;
    private static readonly Regex WordAndSpaceRegex;
    private static readonly ReadOnlyCollection<string> SkipWords;

    static BayesClassifier()
    {
        WordRegex = new Regex( @"[\w]", RegexOptions.Compiled );
        WordAndSpaceRegex = new Regex( @"[^\w\s]", RegexOptions.Compiled );
        SkipWords = new ReadOnlyCollection<string>( new[]
        {
            "a", "again", "all", "along", "are", "also", "an", "and", "as", "at",
            "but", "by", "came", "can", "cant", "couldnt", "did", "didn", "didnt",
            "do", "doesnt", "dont", "ever", "first", "from", "have", "her", "here",
            "him", "how", "i", "if", "in", "into", "is", "isnt", "it", "itll",
            "just", "last", "least", "like", "most", "my", "new", "no", "not", "now",
            "of", "on", "or", "should", "sinc", "so", "some", "th", "than", "this",
            "that", "the", "their", "then", "those", "to", "told", "too", "true",
            "try", "until", "url", "us", "were", "when", "whether", "while", "with",
            "within", "yes", "you", "youll"
        } );
    }

    public static BayesClassifier FromTrainingData( string filePath )
    {
        using( var fileStream = File.Open( filePath, FileMode.Open, FileAccess.Read, FileShare.Read ) )
            return BayesClassifier.FromTrainingData( fileStream );
    }

    public static BayesClassifier FromTrainingData( Stream inputStream )
    {
        var formatter = new BinaryFormatter();
        return formatter.Deserialize( inputStream ) as BayesClassifier;
    }

    private readonly Dictionary<string, Dictionary<string, uint>> TrainingData;
    private readonly Dictionary<string, uint> CategoryWordCounts;
    private readonly Dictionary<string, uint> CategoryCounts;
    private uint TotalWords;

    public ReadOnlyCollection<string> Categories
    {
        get { return this.TrainingData.Keys.ToList().AsReadOnly(); }
    }

    public BayesClassifier( params string[] categoryNames )
    {
        if( categoryNames == null )
            throw new ArgumentNullException( "categoryNames" );

        if( categoryNames.Length < 1 )
            throw new ArgumentException( "Must provide at least one category", "categoryNames" );

        this.TrainingData = new Dictionary<string, Dictionary<string, uint>>();
        this.CategoryWordCounts = new Dictionary<string, uint>();
        this.CategoryCounts = new Dictionary<string, uint>();
        this.TotalWords = 0;

        foreach( var category in categoryNames.Distinct() )
        {
            this.TrainingData.Add( category, new Dictionary<string, uint>() );
            this.CategoryWordCounts.Add( category, 0 );
            this.CategoryCounts.Add( category, 0 );
        }
    }

    public BayesClassifier( SerializationInfo serializationInfo, StreamingContext streamingContext )
    {
        this.TrainingData = serializationInfo.GetValue( "TrainingData", typeof( Dictionary<string, Dictionary<string, uint>> ) ) as Dictionary<string, Dictionary<string, uint>>;
        this.CategoryWordCounts = serializationInfo.GetValue( "WordCounts", typeof( Dictionary<string, uint> ) ) as Dictionary<string, uint>;
        this.CategoryCounts = serializationInfo.GetValue( "Counts", typeof( Dictionary<string, uint> ) ) as Dictionary<string, uint>;
        this.TotalWords = serializationInfo.GetUInt32( "TotalWords" );
    }

    public void GetObjectData( SerializationInfo serializationInfo, StreamingContext streamingContext )
    {
        serializationInfo.AddValue( "TrainingData", this.TrainingData );
        serializationInfo.AddValue( "WordCounts", this.CategoryWordCounts );
        serializationInfo.AddValue( "Counts", this.CategoryCounts );
        serializationInfo.AddValue( "TotalWords", this.TotalWords );
    }

    public void TrainFile( string categoryName, string filePath )
    {
        this.Train( categoryName, new StreamReader( filePath, Encoding.UTF8 ) );
    }

    public void Train( string categoryName, TextReader textReader )
    {
        this.Train( categoryName, textReader.ReadToEnd() );
    }

    public void Train( string categoryName, string text )
    {
        if( categoryName == null )
            throw new ArgumentNullException( "categoryName" );

        if( text == null )
            throw new ArgumentNullException( "text" );

        categoryName = categoryName.ToLowerInvariant();

        if( !this.TrainingData.ContainsKey( categoryName ) )
            throw new KeyNotFoundException( String.Format( "No such category '{0}'", categoryName ) );

        var words = this.StemAndHash( text );
        this.CategoryCounts[categoryName] += 1;

        foreach( var word in words )
        {
            var category = this.TrainingData[categoryName];

            if( category.ContainsKey( word.Key ) )
                category[word.Key] += word.Value;
            else
                category.Add( word.Key, word.Value );

            this.CategoryWordCounts[categoryName] += word.Value;
            this.TotalWords += word.Value;
        }
    }

    public void UntrainFile( string categoryName, string filePath )
    {
        this.Untrain( categoryName, new StreamReader( filePath, Encoding.UTF8 ) );
    }

    public void Untrain( string categoryName, TextReader textReader )
    {
        this.Untrain( categoryName, textReader.ReadToEnd() );
    }

    public void Untrain( string categoryName, string text )
    {
        if( categoryName == null )
            throw new ArgumentNullException( "categoryName" );

        if( text == null )
            throw new ArgumentNullException( "text" );

        categoryName = categoryName.ToLowerInvariant();

        if( !this.TrainingData.ContainsKey( categoryName ) )
            throw new KeyNotFoundException( String.Format( "No such category '{0}'", categoryName ) );

        --this.CategoryCounts[categoryName];

        var words = this.StemAndHash( text );
        foreach( var word in words )
        {
            var count = word.Value;

            if( this.TotalWords >= 0 )
            {
                var category = this.TrainingData[categoryName];
                var originalCount = 0u;

                if( category.ContainsKey( word.Key ) )
                {
                    originalCount = category[word.Key];
                    category[word.Key] -= count;

                    if( category[word.Key] <= 0 )
                    {
                        category.Remove( word.Key );
                        count = originalCount;
                    }
                }

                if( this.CategoryWordCounts[categoryName] >= count )
                    this.CategoryWordCounts[categoryName] -= count;

                this.TotalWords -= count;
            }
        }
    }

    public Dictionary<string, double> ClassificationsFromFile( string filePath )
    {
        return this.Classifications( new StreamReader( filePath, Encoding.UTF8 ) );
    }

    public Dictionary<string, double> Classifications( TextReader textReader )
    {
        return this.Classifications( textReader.ReadToEnd() );
    }

    public Dictionary<string, double> Classifications( string text )
    {
        var scores = new Dictionary<string, double>();
        var hash = this.StemAndHash( text );
        var count = this.CategoryCounts.Values.Sum( x => (double)x );

        foreach( var category in this.TrainingData )
        {
            scores[category.Key] = 0.0;
            var total = this.CategoryWordCounts.ContainsKey( category.Key ) ? this.CategoryWordCounts[category.Key] : 1.0;
            var score = Double.NaN;

            foreach( var word in hash )
            {
                score = category.Value.ContainsKey( word.Key ) ? category.Value[word.Key] : 0.1;
                scores[category.Key] += Math.Log( score / total );
            }

            score = this.CategoryCounts[category.Key] > 0 ? this.CategoryCounts[category.Key] : 0.1;
            scores[category.Key] += Math.Log( score / count );
        }

        return scores;
    }

    public string ClassifyFile( string filePath )
    {
        return this.Classify( new StreamReader( filePath, Encoding.UTF8 ) );
    }

    public string Classify( TextReader textReader )
    {
        return this.Classify( textReader.ReadToEnd() );
    }

    public string Classify( string text )
    {
        return this.Classifications( text )
                   .OrderByDescending( x => x.Value )
                   .First()
                   .Key;
    }

    public void AddCategory( string categoryName )
    {
        if( categoryName == null )
            throw new ArgumentNullException( "categoryName" );

        categoryName = categoryName.ToLowerInvariant();

        if( this.TrainingData.ContainsKey( categoryName ) )
            throw new InvalidOperationException( String.Format( "Category name '{0}' already exists", categoryName ) );

        this.TrainingData.Add( categoryName, new Dictionary<string, uint>() );
    }

    public void SaveTrainingData( string filePath )
    {
        using( var fileStream = File.Open( filePath, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None ) )
        {
            this.SaveTrainingData( fileStream );
            fileStream.Flush();
        }
    }

    public void SaveTrainingData( Stream outputStream )
    {
        var formatter = new BinaryFormatter();
        formatter.Serialize( outputStream, this );
    }

    private Dictionary<string, uint> StemAndHash( string text )
    {
        var words = WordAndSpaceRegex.Replace( text, "" ) .Split( ' ' );
        var symbols = WordRegex.Replace( text, " " ).Split( ' ' );
        var hash = new Dictionary<string, uint>();

        foreach( var word in words )
        {
            var lower = Stemmer.StemWord( word.ToLowerInvariant() );

            if( !SkipWords.Contains( lower ) && lower.Length > 2 )
            {
                if( hash.ContainsKey( lower ) )
                    ++hash[lower];
                else
                    hash.Add( lower, 1 );
            }
        }

        foreach( var symbol in symbols )
        {
            if( hash.ContainsKey( symbol ) )
                ++hash[symbol];
            else
                hash.Add( symbol, 1 );
        }

        return hash;
    }
}