# -*- coding: utf-8 -*-
"""
コンクリート点検アプリ 統合ランチャー
Streamlit無料プランの「プライベートアプリ1個」制限に対応するため、
2つのアプリ(Slab App / Crack2DXF)を1つのアプリとして提供する。

トップのセレクトボックスで使う機能を選ぶと、該当アプリを実行する。
各アプリ本体(apps/slab_app.py, apps/crack2dxf_app.py)は無改変のまま
モジュールとして読み込んで実行する方式。元コードを書き換えないので、
将来それぞれを単体更新しても、このランチャーはそのまま使える。

起動: streamlit run app.py
"""
import os
import sys
import runpy

import streamlit as st

# --- ページ設定はランチャー側で一度だけ行う ---
# (各サブアプリ内の st.set_page_config は、後述の実行ラッパーで無効化する)
st.set_page_config(page_title="コンクリート点検アプリ", layout="wide")

APPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps")
if APPS_DIR not in sys.path:
    sys.path.insert(0, APPS_DIR)

# サブアプリ内の st.set_page_config / st.stop を無害化するためのパッチ
_real_set_page_config = st.set_page_config


def _noop_set_page_config(*args, **kwargs):
    # 2回目以降の set_page_config は Streamlit がエラーにするため握りつぶす
    return None


class _StopAppExecution(Exception):
    """サブアプリ内の st.stop() を、アプリ全体ではなく
    そのサブアプリの実行だけ止めるための例外に置き換える。"""


def _scoped_stop():
    raise _StopAppExecution()


def run_subapp(module_filename: str):
    """apps/ 配下のサブアプリを、その場で実行する。
    set_page_config と stop を一時的に差し替えてから実行し、必ず元に戻す。
    """
    path = os.path.join(APPS_DIR, module_filename)
    st.set_page_config = _noop_set_page_config
    real_stop = st.stop
    st.stop = _scoped_stop
    try:
        runpy.run_path(path, run_name="__main__")
    except _StopAppExecution:
        # サブアプリが「ここで表示を止めたい」と意図した正常な停止
        pass
    finally:
        st.set_page_config = _real_set_page_config
        st.stop = real_stop


# ============================================================
# トップ: 機能選択
# ============================================================
with st.sidebar:
    st.markdown("## 🧰 アプリを選択")
    choice = st.radio(
        "使う機能",
        ["🧱 画像結合 ＆ ゆがみ補正 (Slab App)",
         "🔍 ひび割れ → DXF変換 (Crack2DXF)"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("コンクリート点検支援ツール集")

if choice.startswith("🧱"):
    run_subapp("slab_app.py")
else:
    run_subapp("crack2dxf_app.py")

# 著作権表記(共通フッター)
# ============================================================
COPYRIGHT = "© 2026 [あなたの氏名/組織名]"

st.divider()
st.caption(f"{COPYRIGHT}　無断転載・複製を禁じます。"
           "本ツールは点検補助用です。最終判断は利用者の責任で行ってください。")
