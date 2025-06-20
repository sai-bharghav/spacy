# Regular Expression Characters

This document covers different types of characters used in regular expressions.

## 1. Metacharacters

| Metacharacter | Description | Example |
|---------------|-------------|---------|
| `^`           | Matches the expression to its right at the start of a string. | `^a` matches strings like "aab", "a9c", "apr". |
| `$`           | Matches the expression to its left at the end of a string. | `r$` matches strings like "aaabr", "ar", "r". |
| `.`           | Matches any single character except the line terminator (`\n`). | `b.x` matches "bax", "b9x", "bar". |
| `\|`           | Matches a particular character or group of characters on either side. | `A\|b` matches "a" or "b". |
| `\`           | Escapes a special character. | - |
| `A`           | Matches the character 'A'. | Matches strings like "Amcx", "mnAr", "mnopAx4". |
| `Ab`          | Matches the substring 'Ab'. | Matches strings like "Abcx", "mnAb", "mnopAbx4". |

## 2. Quantifiers

The quantifiers specify the number of occurrences of a character.

| Quantifier | Description | Example |
|------------|-------------|---------|
| `+`        | Matches the expression to its left one or more times. | `s+` matches "s", "ss", "sss", etc. |
| `?`        | Matches the expression to its left 0 or 1 time. | `aS?` matches "a" or "as". |
| `*`        | Matches the expression to its left 0 or more times. | `Br*` matches "B", "Br", "Brr", etc. |
| `{x}`      | Matches the expression to its left exactly x times. | `Mab{5}` matches "Mabbbbb". |
| `{x, }`    | Matches the expression to its left x or more times. | `Xb{3, }` matches "Xbbb", "Xbbbb", etc. |
| `{x,y}`    | Matches the expression to its left at least x times but less than y times. | `Pr{3,6}a` matches "Prrrr" and "Prrrrr". |

## 3. Groups and Ranges

Groups and ranges define a collection of characters enclosed in brackets.

| Group/Range | Description | Example |
|-------------|-------------|---------|
| `( )`       | Matches everything within the simple brackets. | `A(xy)` matches "Axy". |
| `{ }`       | Matches a specific number of occurrences defined in curly brackets. | `xz{4,6}` matches "xzzzzz". |
| `[ ]`       | Matches any character from a range of characters defined in square brackets. | `xz[atp]r` matches "xzar", "xztr", "xzpr". |
| `[pqr]`     | Matches `p`, `q`, or `r` individually. | Matches "p", "q", "r". |
| `[pqr][xy]` | Matches `p`, `q`, or `r`, followed by either `x` or `y`. | Matches "px", "qx", "rx", "py", "qy", "ry". |
| `(?:…)`     | Matches a non-capturing group. | `A(?:nt|pple)` matches "Apple". |
| `[^…..]`    | Matches any character not defined in the square brackets. | `Ab[^pqr]` matches "Ab". |
| `[a-z]`     | Matches lowercase letters from `a` to `z`. | Matches "a", "python", "good". |
| `[A-Z]`     | Matches uppercase letters from `A` to `Z`. | Matches "EXCELLENT", "NATURE". |
| `^[a-zA-Z]` | Matches strings that start with either a lowercase or uppercase letter. | Matches "A854xb", "pv4fv", "cdux". |
| `[0-9]`     | Matches digits from `0` to `9`. | Matches "9845", "54455". |
| `[aeiou]`   | Matches lowercase vowels. | - |
| `[AEIOU]`   | Matches uppercase vowels. | - |
| `ab[^4-9]`  | Matches characters or digits not defined in the square brackets. | Matches strings without "5", "6", "7", "8". |

## 4. Escape Characters or Character Classes

| Character | Description |
|-----------|-------------|
| `\s`      | Matches a single whitespace character. |
| `\S`      | Matches a single non-whitespace character. |
| `\0`      | Matches a NULL character. |
| `\a`      | Matches a bell or alarm. |
| `\d`      | Matches a single decimal digit (0-9). |
| `\D`      | Matches a single non-decimal digit. |
| `\n`      | Matches a newline character. |
| `\w`      | Matches an alphanumeric character (0-9, a-z, A-Z). |
| `\W`      | Matches a single non-word character. |
| `\b`      | Matches a word boundary. |
