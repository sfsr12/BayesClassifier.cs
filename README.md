<h1 align="center">Bayesian Classifier</h1>

A simple implementation of a naive Bayes classifier for textual data.

## Getting started

Download the package uing NuGet or snag one of the archives from the [releases](https://github.com/SirTony/BayesClassifier.cs/releases) page.

    PM> Install-Package BayesClassifier

## Usage

Using the classifier and training it dead-simple.

```csharp
// Quotes from Benjamin Franklin.
var benFranklin = new[]
{
    "Tell me and I forget. Teach me and I remember. Involve me and I learn.",
    "By failing to prepare, you are preparing to fail.",
    "Without continual growth and progress, such words as improvement, achievement, and success have no meaning.",
    "Lost time is never found again.",
    "We are all born ignorant, but one must work hard to remain stupid.",
};

// Quotes from Abraham Lincoln.
var abeLincoln = new[]
{
    "In the end, it's not the years in your life that count. It's the life in your years.",
    "No man has a good enough memory to be a successful liar.",
    "Nearly all men can stand adversity, but if you want to test a man's character, give him power.",
    "The best thing about the future is that it comes one day at a time.",
    "Character is like a tree and reputation like a shadow. The shadow is what we think of it; the tree is the real thing.",
};

// Set up our classifier to train with two categories.
var classifier = new BayesClassifier( "franklin", "lincoln" );

// Train the Franklin quotes.
foreach( var quote in benFranklin )
    classifier.Train( "franklin", quote );

// Train the Lincoln quotes.
foreach( var quote in abeLincoln )
    classifier.Train( "lincoln", quote );

// Attempt to classify a Franklin quote.
Console.WriteLine( classifier.Classify( "Either write something worth reading or do something worth writing." ) );
// Output: franklin

// Attempt to classify a Lincoln quote.
Console.WriteLine( classifier.Classify( "Give me six hours to chop down a tree and I will spend the first four sharpening the axe." ) );
// Output: lincoln
```
