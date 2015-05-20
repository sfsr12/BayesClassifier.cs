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

/// <summary>
/// A naive Bayesian classifier for classifying textual data.
/// </summary>
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

    /// <summary>
    /// Deserializes training data from a file and returns a new pre-trained instance of the classifier.
    /// </summary>
    /// <param name="filePath">The file containing the training data to deserialize.</param>
    /// <returns>A pre-trained instance of <see cref="BayesClassifier"/>, or null if the file does not contain valid training data.</returns>
    public static BayesClassifier FromTrainingData( string filePath )
    {
        using( var fileStream = File.Open( filePath, FileMode.Open, FileAccess.Read, FileShare.Read ) )
            return BayesClassifier.FromTrainingData( fileStream );
    }

    /// <summary>
    /// Deserializes training data from a stream and returns a new pre-trained instance of the classifier.
    /// </summary>
    /// <param name="inputStream">The stream containing the training data to deserialize.</param>
    /// <returns>A pre-trained instance of <see cref="BayesClassifier"/>, or null if the file does not contain valid training data.</returns>
    public static BayesClassifier FromTrainingData( Stream inputStream )
    {
        var formatter = new BinaryFormatter();
        return formatter.Deserialize( inputStream ) as BayesClassifier;
    }

    private readonly Dictionary<string, Dictionary<string, uint>> TrainingData;
    private readonly Dictionary<string, uint> CategoryWordCounts;
    private readonly Dictionary<string, uint> CategoryCounts;
    private uint TotalWords;

    /// <summary>
    /// Gets the category names as a read-only collection.
    /// </summary>
    public ReadOnlyCollection<string> Categories
    {
        get { return this.TrainingData.Keys.ToList().AsReadOnly(); }
    }

    /// <summary>
    /// Initializes a new instance of BayesClassifier with the specified category names.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown if <paramref name="categoryNames"/> is null.</exception>
    /// <exception cref="System.ArgumentException">Thrown if <paramref name="categoryNames"/> does not contain any elements.</exception>
    /// <param name="categoryNames">The categories to use when training the classifier.</param>
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

    /// <summary>
    /// Public constructor for deserializing instances of <see cref="BayesClassifier"/>. Should not be invoked directly.
    /// </summary>
    public BayesClassifier( SerializationInfo serializationInfo, StreamingContext streamingContext )
    {
        this.TrainingData = serializationInfo.GetValue( "TrainingData", typeof( Dictionary<string, Dictionary<string, uint>> ) ) as Dictionary<string, Dictionary<string, uint>>;
        this.CategoryWordCounts = serializationInfo.GetValue( "WordCounts", typeof( Dictionary<string, uint> ) ) as Dictionary<string, uint>;
        this.CategoryCounts = serializationInfo.GetValue( "Counts", typeof( Dictionary<string, uint> ) ) as Dictionary<string, uint>;
        this.TotalWords = serializationInfo.GetUInt32( "TotalWords" );
    }

    /// <summary>
    /// Public method for serializing instances of <see cref="BayesClassifier"/>. Should not be invoked directly.
    /// </summary>
    /// <param name="serializationInfo"></param>
    /// <param name="streamingContext"></param>
    public void GetObjectData( SerializationInfo serializationInfo, StreamingContext streamingContext )
    {
        serializationInfo.AddValue( "TrainingData", this.TrainingData );
        serializationInfo.AddValue( "WordCounts", this.CategoryWordCounts );
        serializationInfo.AddValue( "Counts", this.CategoryCounts );
        serializationInfo.AddValue( "TotalWords", this.TotalWords );
    }

    /// <summary>
    /// Loads a text file and trains the specified category with its contents.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="categoryName"/> or <paramref name="filePath"/> is null.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">Thrown if <paramref name="categoyName"/> does not exist in the classifier's training data.</exception>
    /// <param name="categoryName">The category to train.</param>
    /// <param name="filePath">The file containing the text that will be trained.</param>
    public void TrainFile( string categoryName, string filePath )
    {
        this.Train( categoryName, new StreamReader( filePath, Encoding.UTF8 ) );
    }

    /// <summary>
    /// Reads all text from <paramref name="textReader"/> and trains the specified category with its contents.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="categoryName"/> or <paramref name="text"/> is null.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">Thrown if <paramref name="categoyName"/> does not exist in the classifier's training data.</exception>
    /// <param name="categoryName">The category to train.</param>
    /// <param name="textReader">The <see cref="System.IO.TextReader"/> containing the text that will be trained.</param>
    public void Train( string categoryName, TextReader textReader )
    {
        this.Train( categoryName, textReader.ReadToEnd() );
    }

    /// <summary>
    /// Trains the specified category with the contents of <paramref name="text"/>.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="categoryName"/> or <paramref name="text"/> is null.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">Thrown if <paramref name="categoyName"/> does not exist in the classifier's training data.</exception>
    /// <param name="categoryName">The category to train.</param>
    /// <param name="text">The text that will be trained.</param>
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

    /// <summary>
    /// Loads a text file and un-trains the specified category with its contents.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="categoryName"/> or <paramref name="filePath"/> is null.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">Thrown if <paramref name="categoyName"/> does not exist in the classifier's training data.</exception>
    /// <param name="categoryName">The category to un-train.</param>
    /// <param name="filePath">The file containing the text that will be un-trained.</param>
    public void UntrainFile( string categoryName, string filePath )
    {
        if( filePath == null )
            throw new ArgumentNullException( "filePath" );

        this.Untrain( categoryName, new StreamReader( filePath, Encoding.UTF8 ) );
    }

    /// <summary>
    /// Reads all text from <paramref name="textReader"/> and un-trains the specified category with its contents.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="categoryName"/> or <paramref name="textReader"/> is null.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">Thrown if <paramref name="categoyName"/> does not exist in the classifier's training data.</exception>
    /// <param name="categoryName">The category to un-train.</param>
    /// <param name="filePath">The file containing the text that will be un-trained.</param>
    public void Untrain( string categoryName, TextReader textReader )
    {
        if( textReader == null )
            throw new ArgumentNullException( "textReader" );

        this.Untrain( categoryName, textReader.ReadToEnd() );
    }

    /// <summary>
    /// Un-trains the specified category with the contents of <paramref name="text"/>.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="categoryName"/> or <paramref name="text"/> is null.</exception>
    /// <exception cref="System.Collections.Generic.KeyNotFoundException">Thrown if <paramref name="categoyName"/> does not exist in the classifier's training data.</exception>
    /// <param name="categoryName">The category to un-train.</param>
    /// <param name="filePath">The file containing the text that will be un-trained.</param>
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

    /// <summary>
    /// Loads a file into memory and attempts to classify its contents.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="filePath"/> is null.</exception>
    /// <param name="filePath">The file containing the text to be classified.</param>
    /// <returns>A <see cref="System.Collections.Generic.Dictionary"/> containing the classification probabilities for each category.</returns>
    public Dictionary<string, double> ClassificationsFromFile( string filePath )
    {
        if( filePath == null )
            throw new ArgumentNullException( "filePath" );

        return this.Classifications( new StreamReader( filePath, Encoding.UTF8 ) );
    }

    /// <summary>
    /// Reads all text from <paramref name="textReader"/> and attempts to classify its contents.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="textReader"/> is null.</exception>
    /// <param name="textReader">The <see cref="System.IO.TextReader"/> containing the text to be classified.</param>
    /// <returns>A <see cref="System.Collections.Generic.Dictionary"/> containing the classification probabilities for each category.</returns>
    public Dictionary<string, double> Classifications( TextReader textReader )
    {
        if( textReader == null )
            throw new ArgumentNullException( "textReader" );

        return this.Classifications( textReader.ReadToEnd() );
    }

    /// <summary>
    /// Attempts to classify <paramref name="text"/>.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="text"/> text is null.</exception>
    /// <param name="text">The text to be classified.</param>
    /// <returns>A <see cref="System.Collections.Generic.Dictionary"/> containing the classification probabilities for each category.</returns>
    public Dictionary<string, double> Classifications( string text )
    {
        if( text == null )
            throw new ArgumentNullException( "text" );

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

    /// <summary>
    /// Loads a file into memory and attempts to classify its contents.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="filePath"/> is null.</exception>
    /// <param name="filePath">The file containing the text to be classified.</param>
    /// <returns>The most probably classification for the text.</returns>
    public string ClassifyFile( string filePath )
    {
        return this.Classify( new StreamReader( filePath, Encoding.UTF8 ) );
    }

    /// <summary>
    /// Reads all text from <paramref name="textReader"/> and attempts to classify its contents.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="textReader"/> is null.</exception>
    /// <param name="textReader">The <see cref="System.IO.TextReader"/> containing the text to be classified.</param>
    /// <returns>The most probably classification for the text.</returns>
    public string Classify( TextReader textReader )
    {
        return this.Classify( textReader.ReadToEnd() );
    }

    /// <summary>
    /// Attempts to classify <paramref name="text"/>.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="text"/> text is null.</exception>
    /// <param name="text">The text to be classified.</param>
    /// <returns>The most probably classification for the text.</returns>
    public string Classify( string text )
    {
        return this.Classifications( text )
                   .OrderByDescending( x => x.Value )
                   .First()
                   .Key;
    }

    /// <summary>
    /// Adds a new category to the classifier's training data. This method should be used with care, as it will add new, un-trained category
    /// that will match more text than the more strictly trained, pre-existing categories possible resulting in innacurate classifications.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown if <paramref name="categoryName"/> is null.</exception>
    /// <exception cref="System.InvalidOperationException">Thrown if <paramref name="categoryName"/> already exists in the classifier's training data.</exception>
    /// <param name="categoryName">The new category name to add.</param>
    public void AddCategory( string categoryName )
    {
        if( categoryName == null )
            throw new ArgumentNullException( "categoryName" );

        categoryName = categoryName.ToLowerInvariant();

        if( this.TrainingData.ContainsKey( categoryName ) )
            throw new InvalidOperationException( String.Format( "Category name '{0}' already exists", categoryName ) );

        this.TrainingData.Add( categoryName, new Dictionary<string, uint>() );
    }

    /// <summary>
    /// Serializes the current classifier's training data to a file, allowing it to be re-used later.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="filePath"/> is null.</exception>
    /// <param name="filePath">The file that the training data will be written to.</param>
    public void SaveTrainingData( string filePath )
    {
        if( filePath == null )
            throw new ArgumentNullException( "filePath" );

        using( var fileStream = File.Open( filePath, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None ) )
        {
            this.SaveTrainingData( fileStream );
            fileStream.Flush();
        }
    }

    /// <summary>
    /// Serializes the current classifier's training data to a stream, allowing it to be re-used later.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="outputStream"/> is null.</exception>
    /// <param name="outputStream">The stream that the training data will be written to.</param>
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