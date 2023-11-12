# Regular Expression

`search`
```python
s = 'abcdef'
match = re.search('abc', s) # exactly abc
if match:
    print(1) # 1
```

`sub`
```python
s = 'abcdef'
r = re.sub('[ace]', '', s) # a or c or e
print(r) # bdf
```

`[]`
```python
s = 'abcDEF123'
r = re.sub('[a-zA-Z]', '', s)
print(r) # 123
```

`[^]`
```python
s = 'abcdef'
match = re.search('[^abc]', s) 
# True if there is something that is not a or b or c
if match:
    print(1) # 1
```

`^`: `^` 뒤의 문자열로 문자열이 시작

`$`: `$` 앞의 문자열로 문자열이 끝남
```python
s = 'abcdef'
match = re.search('^abcdef$', s)
if match:
    print(1) # 1
```

`\(`, `\)`
```python
s = 'string(parenthesis)'
r = re.sub(r'\([^)]*\)', '', s)
# r'': raw string
# \(: opening parenthesis
# [^)]*: zero or more characters that are not a closing parenthesis
# \): closing parenthesis
print(r) # string
```

`df.str.replace`
```python
df['col'] = df['col'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', regex=True)
```

`{m}`: m개

`{m, n}`: m개 이상 n개 이하

`{m,}`: m개 이상
```python
s = 'abbbbf'
match = re.search('ab{4}f', s)
if match:
    print(1) # 1
```

`.`: any single character
```python
s = 'abcdef'
match = re.search('a.c', s)
if match:
    print(1) # 1
```

`?`: 0 or 1 occurrences of the pattern to its left

`*`: 0 or more occurrences of the pattern to its left; `{0,}` equivalent

`+`: 1 or more occurrences of the pattern to its left; `{1,}` equivalent
```python
s = 'abcdef'
match = re.search('z?abc', s) # z가 있는 경우 또는 없는 경우 모두
if match:
    print(1) # 1
```

```python
s = 'aaaaaf'
match = re.search('a*f', s) # a가 0개 이상 있는 경우
if match:
    print(1) # 1
```

```python
s = 'abbbbf'
match = re.search('ab+f', s) # b가 1개 이상 있는 경우
if match:
    print(1) # 1
```

`\d`: `[0-9]`와 동일

`\D`: `[^0-9]`와 동일

`\s`: `[ \t\n\r\f\v]`와 동일; whitespace characters (tab, newline, return, form feed, vertical tab)

`\S`: `[^ \t\n\r\f\v]`와 동일

`\w`: `[a-zA-Z0-9_]`와 동일 (`_` 포함)

`\W`: `[^a-zA-Z0-9_]`와 동일 (`_` 포함)

```python
s = 'user_1@gmail.com user.2@gmail.com, user-3@gmail.com'
r = re.findall(r'[\w.-]+@[\w.-]+', s) # . means a literal dot
print(r) # ['user_1@gmail.com', 'user.2@gmail.com', 'user-3@gmail.com']
```

```python
s = 'user_1@gmail.com user.2@gmail.com, user-3@gmail.com'
match = re.search(r'([\w.-]+)@([\w.-]+)', s) # parentheses are added
if match:
    print(match.group())  # user_1@gmail.com
    print(match.group(1)) # user_1
    print(match.group(2)) # gmail.com
```

`findall`
```python
s = 'abc123가나다456'
r = re.findall('\d+', s)
print(r) # ['123', '456']
```

`finditer`
```python
s = 'abc123가나다456'
r = re.finditer('\d+', s)
for i in r:
    print(i.group()) # 123
                     # 456
```

`split`
```python
s = 'abc d ef  gh'
r = re.split('\s+', s)
print(r) # ['abc', 'd', 'ef', 'gh']
```

`()` specifies a capturing group 

`\1` refers to the first matched group
```python
s = 'sentence.'
r = re.sub(r'([?.!,])', r' \1 ', s)
print(r) # sentence . 
```

그냥 ChatGPT에게 물어보자.
