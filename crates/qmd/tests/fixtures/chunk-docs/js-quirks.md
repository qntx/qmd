# JS-quirk parity cases

Documents where JS `\s` differs from Unicode White_Space, and heading
lookahead edge cases.

## Heading runs beyond six hashes

####### seven hashes is not a heading upstream

######## neither is eight

## NEL (U+0085) is not JS whitespace

---
a paragraph continues

-item with NEL separator

1.item with NEL separator

## Whitespace JS does accept

- item with NBSP separator

- item with LINE SEPARATOR separator

-﻿item with FEFF separator

A real space after `---` makes a horizontal rule below:
--- 
## Repeat for chunk pressure

####### again seven hashes

---
-another NEL list line
- normal list line
1.numlist NEL line
2. normal numlist line

More body text to give the chunker enough material to cut around the odd boundary cases above. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

More body text to give the chunker enough material to cut around the odd boundary cases above. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

More body text to give the chunker enough material to cut around the odd boundary cases above. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

More body text to give the chunker enough material to cut around the odd boundary cases above. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

