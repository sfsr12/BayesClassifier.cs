#include "stdafx.hpp"
#include "Stemmer.hpp"

using namespace System::Runtime::InteropServices;

extern "C" {
    struct stemmer;

    struct stemmer* create_stemmer( void );
    void free_stemmer( struct stemmer* z );

    int stem( struct stemmer* z, char* b, int k );
}

String^ Stemmer::StemWord( String^ word )
{
    IntPtr stringPtr = Marshal::StringToHGlobalAnsi( word );
    struct stemmer* z = create_stemmer();

    try
    {
        int newLen = stem( z, static_cast<char*>( stringPtr.ToPointer() ), word->Length );
        return word->Substring( 0, newLen );
    }
    catch( ... )
    {
        throw;
    }
    finally
    {
        Marshal::FreeHGlobal( stringPtr );
        free_stemmer( z );
    }
}