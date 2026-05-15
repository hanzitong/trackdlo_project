#!/bin/bash
# data/raw/ ディレクトリ内で実行すること
# labelme の *_json/ フォルダから label.png を masks/ にコピーする

mkdir -p masks
for d in *_json; do
    cp "$d/label.png" "masks/${d%_json}.png"
done

