# protomol

- `user_guide/*.qmd` are the use-case pages: `>>>` blocks are live doctests (pytest
  `--doctest-glob=*.qmd`), and plain code blocks under a "Planned" callout show API still to
  come. A change that implements a planned call turns its block into a live `>>>` block, and a
  change to a public name or signature updates the pages in the same commit.
