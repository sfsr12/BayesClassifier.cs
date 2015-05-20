#pragma once

using namespace System;

/// <summary>
/// Simple wrapper class to expose the Porter stemming algorithm to .NET
/// </summary>
public ref class Stemmer sealed abstract
{
public:
    /// <summary>
    /// Invokes the Porter stemming algorithm on the given word.
    /// </summary>
    /// <param name="word">The word to stem.</param>
    /// <returns>The newly stemmed word, or an unchanged string if <paramref name="word"/> cannot be stemmed.</returns>
    static String^ StemWord( String^ word );
};
