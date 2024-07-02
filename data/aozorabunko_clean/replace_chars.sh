#!/bin/bash

set -x
temp_dir="temp_dir"
input_basename=$(basename "${1}")
a_modified_file="${temp_dir}/${input_basename}_modified_a.txt"

# has spaces and newlines
b_modified_file="${temp_dir}/${input_basename}_modified_b.txt"

# no spaces or newlines
c_modified_file="${temp_dir}/${input_basename}_modified_c.txt"

cat "$1" | tr '[:upper:]' '[:lower:]' > "$a_modified_file"
cat "$1" | tr '[:upper:]' '[:lower:]' > "$a_modified_file"

## remove double l's
sed -i 's/ll/r/g' "$a_modified_file"

## remove single l's
sed -i 's/l/r/g' "$a_modified_file"

## remove quotes
sed -i "s/'//g" "$a_modified_file"
sed -i 's/"//g' "$a_modified_file"

## remove parenthesis
cat "${a_modified_file}" | tr -d '()' > "$b_modified_file"

sed -i 's/-//g' "$b_modified_file"
sed -i 's_/__g' "$b_modified_file"

# replace special chars * [ ] and +
sed -i 's/*//g' "$b_modified_file"

sed -i 's/\[//g' "$b_modified_file"
sed -i 's/\]//g' "$b_modified_file"

sed -i 's/\!//g' "$b_modified_file"
sed -i 's/#//g' "$b_modified_file"
sed -i 's/%//g' "$b_modified_file"
sed -i 's/+//g' "$b_modified_file"

sed -i 's/\.//g' "$b_modified_file"
sed -i 's/,//g' "$b_modified_file"
sed -i 's/://g' "$b_modified_file"
sed -i 's/;//g' "$b_modified_file"
sed -i 's/>//g' "$b_modified_file"
sed -i 's/<//g' "$b_modified_file"
sed -i 's/=//g' "$b_modified_file"
sed -i 's/?//g' "$b_modified_file"
sed -i 's/\\//g' "$b_modified_file"
sed -i 's/{//g' "$b_modified_file"
sed -i 's/|//g' "$b_modified_file"
sed -i 's/~//g' "$b_modified_file"
sed -i 's/0/zero/g' "$b_modified_file"
sed -i 's/❶/ichi/g' "$b_modified_file"
sed -i 's/❷/ni/g' "$b_modified_file"
sed -i 's/❸/san/g' "$b_modified_file"
sed -i 's/❹/yon/g' "$b_modified_file"
sed -i 's/❺/go/g' "$b_modified_file"
sed -i 's/❻/roku/g' "$b_modified_file"

sed -i 's/はゝ\+/haha/g' "$b_modified_file"
sed -i 's/へゝ\+/hehe/g' "$b_modified_file"
sed -i 's/ほゝ\+/hoho/g' "$b_modified_file"
sed -i 's/ふゝ\+/fufu/g' "$b_modified_file"
sed -i 's/たゝ\+/tata/g' "$b_modified_file"
sed -i 's/だゝ\+/dada/g' "$b_modified_file"
sed -i 's/ずゝ\+/zuzu/g' "$b_modified_file"
sed -i 's/すゝ\+/susu/g' "$b_modified_file"
sed -i 's/つゝ\+/tsutsu/g' "$b_modified_file"
sed -i 's/おゝ\+/oo/g' "$b_modified_file"
sed -i 's/うゝ\+/uu/g' "$b_modified_file"
sed -i 's/らゝ\+/rara/g' "$b_modified_file"
sed -i 's/どゝ\+/dodo/g' "$b_modified_file"
sed -i 's/ゝ\+//g' "$b_modified_file"

# remaining appear to be duplicates for the most part, artifacts of romaji conversion.
# they are also rare in the dataset (less than ~30 occurances each)
# for simplicity removing these to simplify dataset.
sed -i 's/を/ /g' "$b_modified_file"
sed -i 's/ゑ//g' "$b_modified_file"
sed -i 's/あ//g' "$b_modified_file"
sed -i 's/い//g' "$b_modified_file"
sed -i 's/え//g' "$b_modified_file"
sed -i 's/か//g' "$b_modified_file"
sed -i 's/き//g' "$b_modified_file"
sed -i 's/こ//g' "$b_modified_file"
sed -i 's/さ//g' "$b_modified_file"
sed -i 's/し//g' "$b_modified_file"
sed -i 's/す//g' "$b_modified_file"
sed -i 's/ず//g' "$b_modified_file"
sed -i 's/せ//g' "$b_modified_file"
sed -i 's/ち//g' "$b_modified_file"
sed -i 's/ぢ//g' "$b_modified_file"
sed -i 's/づ//g' "$b_modified_file"
sed -i 's/と//g' "$b_modified_file"
sed -i 's/な//g' "$b_modified_file"
sed -i 's/は//g' "$b_modified_file"
sed -i 's/ひ//g' "$b_modified_file"
sed -i 's/へ//g' "$b_modified_file"
sed -i 's/ま//g' "$b_modified_file"
sed -i 's/む//g' "$b_modified_file"
sed -i 's/や//g' "$b_modified_file"
sed -i 's/り//g' "$b_modified_file"
sed -i 's/る//g' "$b_modified_file"
sed -i 's/ゑ//g' "$b_modified_file"

sed -i 's/う//g' "$b_modified_file"
sed -i 's/お//g' "$b_modified_file"
sed -i 's/た//g' "$b_modified_file"
sed -i 's/だ//g' "$b_modified_file"

sed -i 's/つ//g' "$b_modified_file"
sed -i 's/ど//g' "$b_modified_file"
sed -i 's/ふ//g' "$b_modified_file"
sed -i 's/ほ//g' "$b_modified_file"
sed -i 's/よ//g' "$b_modified_file"
sed -i 's/ら//g' "$b_modified_file"

# fu to hu mapping
sed -i 's/fu/hu/g' "$b_modified_file"

sed -i 's/1/ichi/g' "$b_modified_file"
sed -i 's/2/ni/g' "$b_modified_file"
sed -i 's/3/san/g' "$b_modified_file"
sed -i 's/4/yon/g' "$b_modified_file"
sed -i 's/5/go/g' "$b_modified_file"
sed -i 's/6/roku/g' "$b_modified_file"
sed -i 's/7/nana/g' "$b_modified_file"
sed -i 's/8/hachi/g' "$b_modified_file"
sed -i 's/9/kyuu/g' "$b_modified_file"


# # x seems to be a symbol retained, replacing with phonetic approximation
sed -i 's/x/ekusu/g' "$b_modified_file"

# # might be a question mark, removing this:
sed -i 's/q//g' "$b_modified_file"

# replacing v's (only 1901 occurances) with b's for approximation
sed -i 's/v/b/g' "$b_modified_file"

# replacing double consonants with double symbols
sed -i 's/cch/_ch/g' "$b_modified_file"
sed -i 's/ssh/_sh/g' "$b_modified_file"
sed -i 's/ttsu/_tsu/g' "$b_modified_file"
sed -i 's/zz/_z/g' "$b_modified_file"
sed -i 's/pp/_p/g' "$b_modified_file"
sed -i 's/bb/_b/g' "$b_modified_file"
sed -i 's/tt/_t/g' "$b_modified_file"
sed -i 's/kk/_k/g' "$b_modified_file"
sed -i 's/gg/_g/g' "$b_modified_file"
sed -i 's/ff/_f/g' "$b_modified_file"
sed -i 's/rr/_r/g' "$b_modified_file"
sed -i 's/yy/_y/g' "$b_modified_file"
sed -i 's/hh/_h/g' "$b_modified_file"
sed -i 's/jj/_j/g' "$b_modified_file"

sed -i 's/sh/5/g' "$b_modified_file"
sed -i 's/ch/x/g' "$b_modified_file"

# cat "$b_modified_file" | tr -d '\n ' > "$c_modified_file"
cat "$b_modified_file" > "$c_modified_file"

cp "$c_modified_file" input.txt
