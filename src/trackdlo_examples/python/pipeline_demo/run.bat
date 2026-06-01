@echo off
rem pipeline_demo 起動スクリプト (Windows)
rem
rem このスクリプトを pipeline_demo/ に置いて実行する。
rem windows_wheels/ を PYTHONPATH に追加してから pipeline_demo.py を起動する。
rem
rem 前提:
rem   - Python 3.10 がインストール済みであること
rem   - pipeline_demo/ と windows_wheels/ が同じ trackdlo_examples/ 以下にあること
rem
rem ディレクトリ構成:
rem   trackdlo_examples/
rem     windows_wheels/          <- Python パッケージ群
rem     python/pipeline_demo/
rem       run.bat                <- このファイル
rem       pipeline_demo.py
rem       libtrackdlo_c.dll
rem       libpreprocessing_c.dll
rem       libgcc_s_seh-1.dll
rem       libstdc++-6.dll
rem       libwinpthread-1.dll

set SCRIPT_DIR=%~dp0
set TRACKDLO_EXAMPLES_DIR=%SCRIPT_DIR%..\..\

rem Python パッケージの検索パスを設定
set PYTHONPATH=%TRACKDLO_EXAMPLES_DIR%windows_wheels;%PYTHONPATH%

rem torch の DLL (torch_cuda.dll 等) を Windows が見つけられるよう PATH に追加
set PATH=%TRACKDLO_EXAMPLES_DIR%windows_wheels\torch\lib;%PATH%

echo [INFO] PYTHONPATH = %TRACKDLO_EXAMPLES_DIR%windows_wheels
echo [INFO] 起動: pipeline_demo.py

python "%SCRIPT_DIR%pipeline_demo.py" %*
