#!/bin/bash
sudo apt update
sudo apt install mecab libmecab-dev mecab-ipadic-utf8

git clone --depth 1 https://github.com/neologd/mecab-unidic-neologd.git
cd mecab-unidic-neologd
sudo ./bin/install-mecab-unidic-neologd -n -y

echo `mecab-config --dicdir`"/mecab-unidic-neologd"
/usr/local/lib/mecab/dic/mecab-unidic-neologd

echo "dicdir = `mecab-config --dicdir`/mecab-unidic-neologd" | sudo tee /etc/mecabrc
sudo cp /etc/mecabrc /usr/local/etc

python3 -m pip install yakinori
