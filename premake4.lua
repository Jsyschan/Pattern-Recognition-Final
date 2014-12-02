
solution "Final"
  configurations { "Debug", "Release" }

  configuration { "Debug" }
    defines { "_DEBUG", "DEBUG" }
    flags   { "Symbols", "ExtraWarnings" }

  configuration { "Release" }
    defines { "NDEBUG" }
    flags   { "Optimize" }

  platforms { "Native", "x32", "x64", "Universal" }

  project "cluster"

    kind "ConsoleApp"
    language "C++"
    location "example"
    links { "pattern" }

    includedirs { "include" }
    files { "include/*.h", "example/cluster.cpp" }

    configuration { "Debug or Release" }
      targetdir "example/bin"
      objdir "example/obj"

  project "pattern"

    kind "StaticLib"
    language "C++"
    location "src"

    includedirs { "include" }
    files { "include/*.h", "src/*.cpp" }

    configuration { "Debug or Release" }
      targetdir "lib"
      objdir "src/obj"

