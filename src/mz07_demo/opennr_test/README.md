# OpennrTest

## プロジェクト設定

`pyproject.toml`を編集


## 仮想環境作成

`../../uv/uv.exe venv --python ../../python/3.10.11`

その後`.venv/pyenv.cfg`のhomeを絶対パスから相対パスに変更

`home = ../../python/3.10.11`

## パッケージの追加方法

インターネットに繋がっている環境で下記コマンド実行

`../../uv/uv.exe add --link-mode=copy pyxel`

## パッケージの同期

`../../uv/uv.exe sync --link-mode=copy`

## 実行方法

main.pyと同じ場所で下記コマンド実行

`../../uv/uv.exe run main.py`

# OpennrTest

## Project setting

edit `pyproject.toml`

## Create virtual environment

Run:

`../../uv/uv.exe venv --python ../../python/3.10.11`

Then edit `.venv/pyenv.cfg` and change `home` from an absolute path to a relative path:

`home = ../../python/3.10.11`

## How to add a package

On a machine with internet access, run:

`../../uv/uv.exe add --link-mode=copy pyxel`

## Sync packages

`../../uv/uv.exe sync --link-mode=copy`

## How to run

From the same directory as `main.py`, run:

`../../uv/uv.exe run main.py`