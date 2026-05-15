cmake_minimum_required(VERSION 3.14)
project(trackdlo_python)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

# =============================================================================
# [ビルドの全体像]
#
# このパッケージのビルド結果は .so ファイル 1つだけ:
#   trackdlo_python.cpython-310-x86_64-linux-gnu.so
#
# 内部構造:
#   bindings.cpp (pybind11 グルーコード)
#       + libpure_trackdlo.a (アルゴリズム本体、静的リンクで .so に吸収)
#   ↓ コンパイル & リンク
#   trackdlo_python.cpython-310-...so  ← Python が import するファイル
#
# Python 側は `import trackdlo_python` で .so をロードするだけでよい。
# pure_trackdlo の .a は .so の中に静的リンクされており、
# Python からは直接見えない (見える必要もない)。
# =============================================================================

# =============================================================================
# [Python の選択]
#
# find_program は PATH を先頭から検索し、最初に見つかった python3 を使う。
# この環境では /usr/bin/python3 (3.10.12) が最初に見つかる。
#
# なぜ明示するか:
#   uv が管理する Python 3.11 が PATH に混入することがある。
#   pybind11 が使う Python と cmake が使う Python がずれると
#   バージョン不一致でビルドが壊れる。
#   ここで一度 PYTHON3_EXE に固定しておくことで、
#   以降の execute_process / find_package が同じ Python を参照する。
#
# PYTHON_EXECUTABLE は pybind11 の旧 CMake モジュール (FindPythonLibsNew.cmake)
# が参照するキャッシュ変数。pybind11 2.x 系では必須。3.x 系では
# find_package(Python3) に移行しているが、互換のため残す。
# =============================================================================
find_program(PYTHON3_EXE python3 REQUIRED)
set(PYTHON_EXECUTABLE "${PYTHON3_EXE}")

# =============================================================================
# [pybind11 の選択]
#
# system にインストールされた pybind11 (2.9.1) と
# pip でインストールされた pybind11 (3.0.x) が共存している。
#
# system 2.9.1 を使えない理由:
#   type_caster_base.h 内で `frame->f_back` を参照しているが、
#   Python 3.11 以降は PyFrameObject が不透明型 (opaque type) になり
#   フィールドに直接アクセスできなくなった。
#   → コンパイルエラー: "invalid use of incomplete type 'PyFrameObject'"
#
# 解決策: pip の pybind11 3.0.x を優先する。
#
# execute_process は cmake の configure 時 (cmake .. を実行した瞬間) に走る。
# python3 に pybind11 がインストールされていれば get_cmake_dir() が
# そのパッケージの cmake/ ディレクトリを返す。
# uv で作った .venv を activate してから colcon build すると
# .venv の pybind11 のパスが返るため、環境を切り替えても自動追従する。
#
# find_package の HINTS: CMAKE_PREFIX_PATH より高優先でパスを探す。
# ここで渡すことで system の /usr/lib/cmake/pybind11/ より先に
# pip のパスが検索される。
# =============================================================================
execute_process(
    COMMAND "${PYTHON3_EXE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_STATUS
)
if(NOT PYBIND11_STATUS EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. 'uv sync' または 'pip install pybind11' を実行してください")
endif()
find_package(pybind11 REQUIRED HINTS "${PYBIND11_CMAKE_DIR}")

# pure_trackdlo は colcon が先にビルドして install/ 以下に置く。
# CMAKE_PREFIX_PATH に install/ が含まれるため find_package で見つかる。
# (package.xml の <depend>pure_trackdlo</depend> がビルド順を保証する)
find_package(pure_trackdlo REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# =============================================================================
# [pybind11_add_module]
#
# 通常の add_library(... SHARED) ではなくこのマクロを使う理由:
#   1. ファイル名を自動で決める
#      → trackdlo_python.cpython-310-x86_64-linux-gnu.so のように
#        Python バージョンとアーキテクチャを含む名前になる。
#        Python の import 機構はこの命名規則でファイルを探す。
#   2. 必要なコンパイルフラグを自動付与する
#      -fvisibility=hidden : .so が外部に公開するシンボルを pybind11 の
#        エントリポイントだけに絞る。これにより他の .so との
#        シンボル衝突を防ぐ。
#      -fPIC は pure_trackdlo 側 (POSITION_INDEPENDENT_CODE ON) で
#        設定済み。静的ライブラリを .so に組み込む際に必要。
#   3. Python のインクルードパスとリンクフラグを自動設定する
#      手動で target_include_directories に Python のパスを書く必要がない。
#
# PRIVATE: pure_trackdlo と Eigen3 は trackdlo_python の実装詳細であり、
#   この .so を使う側 (Python) には伝播させない。
# =============================================================================
pybind11_add_module(trackdlo_python src/bindings.cpp)
target_link_libraries(trackdlo_python PRIVATE pure_trackdlo Eigen3::Eigen)
