# FILE

## Ideas

- The pattern `structure | fixs<'x', 'y'>(10, 20) | get_at(data)` is too common, there should exist an abstraction for that
  - possible solution: `structure | get_fixs<'x', 'y'>(data, 10, 20)`
