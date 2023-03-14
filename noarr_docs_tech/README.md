# Technical documentation for Noarr Structures

## Structure

A  *structure* is a simple object that describes data layouts and their abstractions

### Subtypes of structures

- **Cube:** a cube is a structure hierarchy that has all its dimensions dynamic. This has a consequence of having a single scalar type (all values described by the structure share the same type).

- **Point:** a point is a structure hierarchy with no dimensions. It has only a single scalar type and it describes one scalar value of this type.

  It is a special case of a cube.
