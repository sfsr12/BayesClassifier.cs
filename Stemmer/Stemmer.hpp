#pragma once

using namespace System;

public ref class Stemmer sealed abstract
{
public:
    static String^ StemWord( String^ word );
};
