#include "stdafx.hpp"

using namespace System;
using namespace System::Reflection;
using namespace System::Runtime::CompilerServices;
using namespace System::Runtime::InteropServices;
using namespace System::Security::Permissions;

[assembly:AssemblyTitleAttribute( L"Stemmer" )];
[assembly:AssemblyDescriptionAttribute( L"A .NET wrapper for the Porter stemmer algorithm" )];

#ifdef _DEBUG
    [assembly:AssemblyConfigurationAttribute( L"Debug" )];
#else
    [assembly:AssemblyConfigurationAttribute( L"Release" )];
#endif

[assembly:AssemblyCompanyAttribute( L"" )];
[assembly:AssemblyProductAttribute( L"Stemmer" )];
[assembly:AssemblyCopyrightAttribute( L"Copyright (c) Tony J. Hudgins 2015" )];
[assembly:AssemblyTrademarkAttribute( L"" )];
[assembly:AssemblyCultureAttribute( L"" )];
[assembly:AssemblyVersionAttribute( "1.0.2.3" )];
[assembly:ComVisible( false )];
[assembly:CLSCompliantAttribute( true )];