# UTF-8 and URL encoding

## Quick example: 

Hex string from the body of HTTP POST request:
```
6e 61 6d 65 25 33 44 25 45 41 25 42 39 25 38 30 25 45 42 25 42 33 25 38 34 25 45 42 25 42 39 25 39 44 25 32 36 70 68 6f 6e 65 25 33 44 30 31 30 37 38 32 37 32 34 38 37 25 32 36 62 69 72 74 68 64 61 79 25 33 44 32 33 30 35 32 38 25 32 36 6b 65 79 5f 6e 75 6d 25 33 44 59 32 56 6e 56 6c 42 4d 55 47 4a 70 64 46 64 58 54 48 49 33 53 47 46 47 54 6e 68 33 51 54 30 39 
```

...converted into ASCII:
```
name%3D%EA%B9%80%EB%B3%84%EB%B9%9D%26phone%3D01078272487%26birthday%3D230528%26key_num%3DY2VnVlBMUGJpdFdXTHI3SGFGTnh3QT09
```

...with URL-encoded parts decoded:
```
name=김별빝&phone=01078272487&birthday=230528&key_num=Y2VnVlBMUGJpdFdXTHI3SGFGTnh3QT09
```

## URL encoding - why?
URLs can only be sent over the internet using the ASCII character set. But data often contains characters outside the ASCII set, or characters that are not safe to include in a URL because they can be confused with URL control characters, such as the spaces and special characters like ", <, >, #, {, }, |, \, ^, ~, \[, \], and \`. URL encoding replaces unsafe ASCII characters with a "%" followed by two hexadecimal digits corresponding to the character values in the UTF-8 character set. For example, the space character " " would be replaced with "%20".

## Unicode Transformation Format, UTF
ASCII characters are represented as single bytes, the same as in ASCII. Characters outside the ASCII range are represented as sequences of 2, 3, or 4 bytes. Most Latin-based characters with diacritics and Greek characters are encoded as two bytes. Cyrillic, Hebrew, Arabic, and many other scripts also only require two bytes. Many Asian scripts, including Chinese, Japanese, and Korean, are encoded as three bytes. Emoji and many less common scripts require four bytes.

In UTF-8, Korean characters are represented by three bytes. The specific encoding of '김' as EA B9 80 in UTF-8 is dictated by the Unicode standard. Unicode assigns a unique number, known as a code point, to each character in almost all of the world's writing systems. '김' has the Unicode code point U+AE40. The UTF-8 encoding of this code point is the byte sequence EA B9 80. 

## Inner workings of the transformation:

### How many bytes should it take? - templates for each possible length
- 1 byte: Used for ASCII characters (U+0000 to U+007F). Template: `0xxxxxxx`
- 2 bytes: Used for characters from U+0080 to U+07FF. Template: `110xxxxx 10xxxxxx`
- 3 bytes: Used for characters from U+0800 to U+FFFF. Template: `1110xxxx 10xxxxxx 10xxxxxx`
- 4 bytes: Used for characters from U+10000 to U+10FFFF. Template: `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`

### ...which is how we know how many bytes make up a single character
You can look at the first few bits of the first hexadecimal digit to recognize whether a % followed by two hexadecimal digits is part of a multi-byte sequence. UTF-8 uses these bits to indicate the start of a new character and the number of bytes in the character:
- If the first bits are `0`, it's a single-byte (ASCII) character.
- If the first bits are `110`, it's the start of a two-byte character.
- If the first bits are `1110`, it's the start of a three-byte character.
- If the first bits are `11110`, it's the start of a four-byte character.

### Encoding steps:
1. Start with a character, for instance, `김`.
2. It corresponds to `U+AE40` in Unicode.
3. Translate this into binary, getting `10101110 01000000`.
4. Since `U+AE40` falls between `U+0800` and `U+FFFF`, it gets encoded into three bytes in UTF-8, following the template: `1110xxxx 10xxxxxx 10xxxxxx`.
5. So we get `1110 1010 10 111001 10 000000` from steps 3 and 4.
6. Convert these bytes to hexadecimal, resulting in `EA B9 80`.
7. And finally, URL encode it as `%EA%B9%80`.

## Python code:
```python
import urllib.parse

hex_str = "6e 61 6d 65 25 33 44 25 45 41 25 42 39 25 38 30 25 45 42 25 42 33 25 38 34 25 45 42 25 42 39 25 39 44 25 32 36 70 68 6f 6e 65 25 33 44 30 31 30 37 38 32 37 32 34 38 37 25 32 36 62 69 72 74 68 64 61 79 25 33 44 32 33 30 35 32 38 25 32 36 6b 65 79 5f 6e 75 6d 25 33 44 59 32 56 6e 56 6c 42 4d 55 47 4a 70 64 46 64 58 54 48 49 33 53 47 46 47 54 6e 68 33 51 54 30 39"
text = bytes.fromhex(hex_str).decode('utf-8')
text = urllib.parse.unquote(text)

print(text)
```
```
name=김별빝&phone=01078272487&birthday=230528&key_num=Y2VnVlBMUGJpdFdXTHI3SGFGTnh3QT09
```
