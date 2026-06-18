#!/bin/bash
# data/raw/ ディレクトリ内で実行すること
# labelme の .json ファイルを *_json/ フォルダに展開する

for f in *json; do
    python3 -m labelme.cli.json_to_dataset "$f"
done


