# Developer documentation for Noarr Structures

This directory contains sections that are interesting mostly or exclusively to Noarr developers.

Most of the library's functionality is in the structure classes and shortcuts for these.
The library allows clients to [define their own structures](../DefiningStructures.md#defining-structures-manually),
so it has to give them the same interfaces it itself uses internally (e.g. [signatures](../Signature.md)).

Currently, there are just two topics that even the most advanced clients can ignore:

- the style and other [conventions](Conventions.md) that are to be followed by the library code, but does not apply to client code
- the [`noarr::contain`](Contain.md) class, from which structures usually inherit, but usually are not required to
