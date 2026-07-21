# -*- coding: utf-8 -*-
"""
ひび割れマーキング写真 → DXF変換アプリ v2 (Streamlit)
対応マーカー: 赤・緑・青・黒マジック、白チョーク

起動方法:
    pip install streamlit opencv-python-headless scikit-image ezdxf numpy
    streamlit run crack2dxf_app.py
"""
import io
import os
import tempfile
import zipfile

import cv2
import numpy as np
import streamlit as st
from skimage.morphology import skeletonize
import ezdxf
from ezdxf.addons import odafc

st.set_page_config(page_title="ひび割れ写真→DXF変換 v2", layout="wide")

# ======================================================================
# GA4 アクセス解析（全体の実行回数・再訪ユーザー数の計測）
# ======================================================================
import streamlit.components.v1 as components

GA_ID = "G-JGKS7NWTZD"  # ← GA4 測定ID


def inject_ga():
    """GA4 本体を親ドキュメントに1回だけ注入（page_view 自動計測）"""
    components.html(f"""
    <script>
    (function() {{
      var doc = window.parent.document;
      if (doc.getElementById('ga-lib')) return;   // 二重注入防止
      var s = doc.createElement('script');
      s.id = 'ga-lib'; s.async = true;
      s.src = 'https://www.googletagmanager.com/gtag/js?id={GA_ID}';
      doc.head.appendChild(s);
      var s2 = doc.createElement('script');
      s2.innerHTML = "window.dataLayer=window.dataLayer||[];"
        + "function gtag(){{dataLayer.push(arguments);}}"
        + "gtag('js', new Date());"
        + "gtag('config','{GA_ID}');";
      doc.head.appendChild(s2);
    }})();
    </script>
    """, height=0)


def track_event(event_name):
    """任意イベントを GA4 に送信"""
    components.html(f"""
    <script>
    (function() {{
      var g = window.parent.gtag;
      if (g) {{ g('event', '{event_name}'); }}
    }})();
    </script>
    """, height=0)


inject_ga()
# ======================================================================

# ----------------------------------------------------------------------
# 画像処理関数
# ----------------------------------------------------------------------

def clean_small(mask, min_area):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


def extract_red(img, s_min, v_min, hue_width, min_area):
    """赤マジック (HSV 0付近の2領域)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, s_min, v_min]),
                     np.array([hue_width, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([180 - hue_width, s_min, v_min]),
                     np.array([180, 255, 255]))
    return clean_small(m1 | m2, min_area)


def extract_hue_band(img, h_lo, h_hi, s_min, v_min, min_area):
    """緑・青など 色相帯指定の汎用抽出"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([h_lo, s_min, v_min]),
                    np.array([h_hi, 255, 255]))
    return clean_small(m, min_area)


def extract_black(img, exclude_mask, v_max, bh_thresh, max_width, min_span,
                  remove_h_rule, remove_v_rule):
    """黒マジック (ブラックハット + 線らしさフィルタ)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k15)
    cand = ((blackhat > bh_thresh) & (gray < v_max)
            & (hsv[:, :, 1] < 110)).astype(np.uint8) * 255

    margin = max(10, int(min(h, w) * 0.015))
    border = np.zeros_like(cand)
    border[margin:h - margin, margin:w - margin] = 255
    cand &= border
    cand[exclude_mask > 0] = 0

    if remove_h_rule:
        hor = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1)))
        cand[cv2.dilate(hor, np.ones((3, 3), np.uint8)) > 0] = 0
    if remove_v_rule:
        ver = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)))
        cand[cv2.dilate(ver, np.ones((3, 3), np.uint8)) > 0] = 0

    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                            iterations=2)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    out = np.zeros_like(cand)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 110:
            continue
        comp = (lab == i).astype(np.uint8)
        sk = skeletonize(comp > 0)
        L = sk.sum()
        if L < 50:
            continue
        width = area / max(L, 1)
        span = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if width > max_width or span < min_span:
            continue
        ys, xs = np.where(comp > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        p = pts - pts.mean(0)
        evals, evecs = np.linalg.eigh((p.T @ p) / len(p))
        straightness = float(np.sqrt(evals[0]))
        angle = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))) % 180
        if min(abs(angle), abs(angle - 180), abs(angle - 90)) < 3 and straightness < 2.0:
            continue
        out[lab == i] = 255
    return out


def extract_chalk(img, exclude_mask, v_min, th_thresh, s_max,
                  w_min, w_max, max_cv, min_span):
    """白チョーク抽出 + エフロレッセンス識別
    識別原理: チョーク線は人が引いた線なので線幅がほぼ一定 (幅の変動係数CVが小さい)。
    エフロは斑状・筋幅不規則 → CV大 or 幅過大として棄却し、reviewマスクに回す。
    戻り値: (chalk_mask, rejected_mask=エフロ等と判定した白色領域)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k15)
    cand = ((tophat > th_thresh) & (gray > v_min)
            & (hsv[:, :, 1] < s_max)).astype(np.uint8) * 255

    margin = max(10, int(min(h, w) * 0.015))
    border = np.zeros_like(cand)
    border[margin:h - margin, margin:w - margin] = 255
    cand &= border
    cand[exclude_mask > 0] = 0
    # 注意: ここでcloseを強くかけると白斑が巨大成分に融合してチョーク線を
    # 飲み込むため、closeは最小限(3x3 1回)にとどめる
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    chalk = np.zeros_like(cand)
    rejected = np.zeros_like(cand)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 80:
            continue
        comp = (lab == i).astype(np.uint8)
        sk = skeletonize(comp > 0)
        L = sk.sum()
        span = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if L < 40 or span < min_span:
            rejected[lab == i] = 255
            continue
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        widths = 2.0 * dist[sk]
        mean_w = float(widths.mean())
        cv_w = float(widths.std()) / max(mean_w, 1e-6)
        if (w_min <= mean_w <= w_max) and cv_w <= max_cv:
            chalk[lab == i] = 255          # 線幅一定 → チョーク線
        else:
            rejected[lab == i] = 255        # 幅不規則/過大 → エフロ等
    return chalk, rejected


def extract_color_chalk(img, exclude_mask, hue_ranges, s_min, v_min,
                        w_min, w_max, max_cv, min_span):
    """カラーチョーク抽出 (ピンク・青・赤・黄など 有彩色チョーク)
    白チョークと同じく『線幅が一定』という特徴で線らしさを判定し、
    斑状・不規則な有彩色領域 (汚れ・塗膜剥離など) を棄却する。
    hue_ranges: [(h_lo, h_hi), ...] 色相の許容帯 (赤など0跨ぎは2帯で渡す)
    戻り値: (chalk_mask, rejected_mask)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    cand = np.zeros((h, w), np.uint8)
    for h_lo, h_hi in hue_ranges:
        cand |= cv2.inRange(hsv, np.array([h_lo, s_min, v_min]),
                            np.array([h_hi, 255, 255]))

    margin = max(10, int(min(h, w) * 0.015))
    border = np.zeros_like(cand)
    border[margin:h - margin, margin:w - margin] = 255
    cand &= border
    cand[exclude_mask > 0] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    chalk = np.zeros_like(cand)
    rejected = np.zeros_like(cand)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 80:
            continue
        comp = (lab == i).astype(np.uint8)
        sk = skeletonize(comp > 0)
        L = sk.sum()
        span = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if L < 40 or span < min_span:
            rejected[lab == i] = 255
            continue
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        widths = 2.0 * dist[sk]
        mean_w = float(widths.mean())
        cv_w = float(widths.std()) / max(mean_w, 1e-6)
        if (w_min <= mean_w <= w_max) and cv_w <= max_cv:
            chalk[lab == i] = 255          # 線幅一定 → 色チョーク線
        else:
            rejected[lab == i] = 255        # 幅不規則 → 汚れ等
    return chalk, rejected


# 色チョークの既定色相帯 (OpenCV HSV: H=0-179)
COLOR_CHALK_DEFS = {
    "pink":   {"jp": "ピンク", "draw": (203, 192, 255), "dxf": 6,
               "hue": [(150, 175)], "s_min": 40, "v_min": 120},
    "blue":   {"jp": "青",     "draw": (255, 100, 0),   "dxf": 5,
               "hue": [(95, 130)],  "s_min": 50, "v_min": 80},
    "red":    {"jp": "赤",     "draw": (0, 0, 255),     "dxf": 1,
               "hue": [(0, 10), (170, 180)], "s_min": 60, "v_min": 80},
    "yellow": {"jp": "黄",     "draw": (0, 255, 255),   "dxf": 2,
               "hue": [(20, 35)],  "s_min": 50, "v_min": 120},
}


def trace_skeleton(mask):
    """スケルトン画像 → 画素パス列 (分岐点・端点で分割)"""
    sk = skeletonize(mask > 0)
    pts = set(zip(*np.where(sk)))
    nbrs8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def neighbors(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in nbrs8
                if (p[0] + dy, p[1] + dx) in pts]

    deg = {p: len(neighbors(p)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}
    visited, paths = set(), []
    for n0 in nodes:
        for nb in neighbors(n0):
            if (n0, nb) in visited:
                continue
            path = [n0, nb]
            visited.add((n0, nb)); visited.add((nb, n0))
            prev, cur = n0, nb
            while cur not in nodes:
                nxt = [q for q in neighbors(cur)
                       if q != prev and (cur, q) not in visited]
                if not nxt:
                    break
                q = nxt[0]
                visited.add((cur, q)); visited.add((q, cur))
                path.append(q)
                prev, cur = cur, q
            paths.append(path)
    looped = pts - {p for path in paths for p in path}
    while looped:
        start = next(iter(looped))
        path = [start]; looped.discard(start)
        prev, cur = None, start
        while True:
            nxt = [q for q in neighbors(cur) if q != prev and q in looped]
            if not nxt:
                break
            q = nxt[0]; path.append(q); looped.discard(q)
            prev, cur = cur, q
        if len(path) > 5:
            paths.append(path)
    return paths


PHOTO_FILENAME = "crack_photo.png"


def build_dxf(layer_masks, img_shape, real_w, real_h, eps, min_len_mm,
              embed_photo=False):
    """{レイヤ名: (mask, dxf色番号)} → (DXFバイト列, ezdxfドキュメント, 統計, px座標ポリライン)
    embed_photo=True のとき、PHOTOレイヤに写真(外部参照 crack_photo.png)を
    床版実寸に合わせて下絵として配置する。
    """
    H, W = img_shape[:2]
    sx, sy = real_w / W, real_h / H

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    if embed_photo:
        # 写真を下絵として配置 (同フォルダの crack_photo.png を相対参照)
        doc.layers.add("PHOTO", color=8)
        image_def = doc.add_image_def(filename=PHOTO_FILENAME,
                                      size_in_pixel=(W, H))
        msp.add_image(image_def=image_def, insert=(0, 0),
                      size_in_units=(real_w, real_h), rotation=0,
                      dxfattribs={"layer": "PHOTO"})
    doc.layers.add("OUTLINE", color=8)
    msp.add_lwpolyline([(0, 0), (real_w, 0), (real_w, real_h), (0, real_h)],
                       close=True, dxfattribs={"layer": "OUTLINE"})

    stats, px_polys = {}, {}
    for layer, (mask, color) in layer_masks.items():
        doc.layers.add(layer, color=color)
        px_polys[layer] = []
        count, total = 0, 0.0
        if mask is not None:
            for path in trace_skeleton(mask):
                if len(path) < 6:
                    continue
                arr = np.array([[p[1], p[0]] for p in path],
                               np.float32).reshape(-1, 1, 2)
                pl = cv2.approxPolyDP(arr, eps, False).reshape(-1, 2)
                if len(pl) < 2:
                    continue
                mm = [(x * sx, real_h - y * sy) for x, y in pl]
                seglen = sum(np.hypot(mm[i + 1][0] - mm[i][0],
                                      mm[i + 1][1] - mm[i][1])
                             for i in range(len(mm) - 1))
                if seglen < min_len_mm:
                    continue
                msp.add_lwpolyline(mm, dxfattribs={"layer": layer})
                px_polys[layer].append(pl.astype(int))
                count += 1
                total += seglen
        stats[layer] = (count, total)

    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        doc.saveas(f.name)
        with open(f.name, "rb") as g:
            data = g.read()
    return data, doc, stats, px_polys


def export_dwg_bytes(doc, oda_path=""):
    """ezdxfドキュメント → DWGバイト列 (ODA File Converter が必要)
    戻り値: (dwg_bytes | None, エラーメッセージ | None)
    """
    if oda_path:
        ezdxf.options.set("odafc-addon", "win_exec_path", oda_path)
    try:
        if not odafc.is_installed():
            return None, (
                "ODA File Converter が見つかりません。"
                "https://www.opendesign.com/guestfiles/oda_file_converter "
                "から無償版をインストールするか、サイドバーでexeのパスを指定してください。")
        with tempfile.TemporaryDirectory() as td:
            dwg_path = os.path.join(td, "crack_map.dwg")
            odafc.export_dwg(doc, dwg_path, version="R2018", replace=True)
            with open(dwg_path, "rb") as f:
                return f.read(), None
    except Exception as e:
        return None, f"DWG変換に失敗しました: {e}"


def make_cad_zip(cad_bytes, cad_name, photo_png_bytes):
    """CADファイル + 参照画像を1つのZIPにまとめる"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(cad_name, cad_bytes)
        z.writestr(PHOTO_FILENAME, photo_png_bytes)
    return buf.getvalue()


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("🔍 ひび割れマーキング写真 → DXF変換 v2")
st.caption("赤・緑・青・黒マジックおよび白チョークのマーキングを抽出し、"
           "実寸スケールのポリラインとしてDXF出力します。")

with st.sidebar:
    st.header("⚙️ パラメータ")

    st.subheader("実寸法")
    real_w = st.number_input("幅 (mm)", 100.0, 100000.0, 2000.0, 100.0)
    real_h = st.number_input("高さ (mm)", 100.0, 100000.0, 1500.0, 100.0)

    st.subheader("抽出する色")
    use_red = st.checkbox("🔴 赤マジック", True)
    use_green = st.checkbox("🟢 緑マジック", False)
    use_blue = st.checkbox("🔵 青マジック", False)
    use_black = st.checkbox("⚫ 黒マジック", True)
    use_chalk = st.checkbox("⚪ 白チョーク", False)
    st.markdown("**🖍 カラーチョーク**")
    use_cc = {
        "pink":   st.checkbox("🩷 ピンクチョーク", False),
        "blue":   st.checkbox("🔵 青チョーク", False),
        "red":    st.checkbox("🔴 赤チョーク", False),
        "yellow": st.checkbox("🟡 黄チョーク", False),
    }

    if use_red:
        with st.expander("🔴 赤の設定"):
            r_smin = st.slider("彩度しきい値", 20, 150, 60, key="rs")
            r_vmin = st.slider("明度しきい値", 20, 150, 60, key="rv")
            r_hue = st.slider("色相幅 (±H)", 5, 25, 12, key="rh")
            r_minarea = st.slider("最小面積 (px)", 5, 200, 25, key="ra")
    if use_green:
        with st.expander("🟢 緑の設定"):
            g_hlo = st.slider("色相 下限", 30, 60, 38, key="ghl")
            g_hhi = st.slider("色相 上限", 60, 95, 85, key="ghh")
            g_smin = st.slider("彩度しきい値", 20, 150, 60, key="gs")
            g_vmin = st.slider("明度しきい値", 20, 150, 60, key="gv")
            g_minarea = st.slider("最小面積 (px)", 5, 200, 25, key="ga")
    if use_blue:
        with st.expander("🔵 青の設定"):
            b_hlo = st.slider("色相 下限", 85, 105, 95, key="bhl")
            b_hhi = st.slider("色相 上限", 105, 140, 130, key="bhh")
            bl_smin = st.slider("彩度しきい値", 20, 150, 60, key="bs")
            bl_vmin = st.slider("明度しきい値", 20, 150, 60, key="bv")
            bl_minarea = st.slider("最小面積 (px)", 5, 200, 25, key="ba")
    if use_black:
        with st.expander("⚫ 黒の設定"):
            k_vmax = st.slider("暗さしきい値 (V max)", 60, 180, 110, key="kv")
            k_bh = st.slider("ブラックハット強度", 20, 80, 40, key="kb")
            k_width = st.slider("最大線幅 (px)", 3.0, 15.0, 6.5, 0.5, key="kw")
            k_span = st.slider("最小スパン (px)", 20, 150, 60, key="ks")
            rule_h = st.checkbox("水平罫線を除去", True, key="krh")
            rule_v = st.checkbox("垂直罫線を除去", False, key="krv")
    if use_chalk:
        with st.expander("⚪ 白チョークの設定"):
            c_vmin = st.slider("明るさしきい値 (V min)", 120, 230, 170, key="cv")
            c_th = st.slider("トップハット強度", 20, 90, 50, key="ct")
            c_smax = st.slider("彩度上限", 30, 100, 60, key="cs")
            c_wmin, c_wmax = st.slider("線幅範囲 (px)", 1.0, 20.0, (2.0, 12.0),
                                       0.5, key="cw")
            c_cv = st.slider("線幅の変動係数 上限", 0.2, 0.8, 0.45, 0.05, key="ccv",
                             help="小さいほど『幅が一定の線』だけをチョークと判定。"
                                  "エフロレッセンスは幅が不規則なので棄却される")
            c_span = st.slider("最小スパン (px)", 20, 150, 50, key="csp")
            show_rejected = st.checkbox("棄却領域(エフロ等)を表示", True, key="crj")

    cc_params = {}
    for ck, on in use_cc.items():
        if not on:
            continue
        d = COLOR_CHALK_DEFS[ck]
        with st.expander(f"🖍 {d['jp']}チョークの設定"):
            s_min = st.slider("彩度しきい値", 10, 150, d["s_min"], key=f"cc_s_{ck}")
            v_min = st.slider("明度しきい値", 30, 200, d["v_min"], key=f"cc_v_{ck}")
            wmin, wmax = st.slider("線幅範囲 (px)", 1.0, 20.0, (2.0, 12.0),
                                   0.5, key=f"cc_w_{ck}")
            cvmax = st.slider("線幅の変動係数 上限", 0.2, 0.8, 0.45, 0.05,
                              key=f"cc_cv_{ck}",
                              help="小さいほど『幅が一定の線』だけをチョークと判定")
            span = st.slider("最小スパン (px)", 20, 150, 50, key=f"cc_sp_{ck}")
            # 赤チョークは色相0跨ぎ。微調整スライダは中心幅で提供
            hue_ranges = d["hue"]
            if ck != "red":
                hc = int((d["hue"][0][0] + d["hue"][0][1]) / 2)
                hw = st.slider("色相中心 ±幅", 5, 25,
                               int((d["hue"][0][1] - d["hue"][0][0]) / 2),
                               key=f"cc_h_{ck}")
                hcc = st.slider("色相中心", max(0, hc - 30), min(179, hc + 30),
                                hc, key=f"cc_hc_{ck}")
                hue_ranges = [(max(0, hcc - hw), min(179, hcc + hw))]
            cc_params[ck] = dict(hue=hue_ranges, s_min=s_min, v_min=v_min,
                                 wmin=wmin, wmax=wmax, cvmax=cvmax, span=span)

    st.subheader("出力設定")
    embed_photo = st.checkbox("📷 写真を下絵として埋め込む", True,
                              help="CADファイルと同フォルダの crack_photo.png を"
                                   "外部参照します(ZIPで一括ダウンロード)")
    make_dwg = st.checkbox("📐 DWGファイルも作成", False,
                           help="無償の ODA File Converter のインストールが必要です")
    oda_path = ""
    if make_dwg:
        oda_path = st.text_input(
            "ODA File Converter のパス (空欄=自動検出)",
            value="",
            placeholder=r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe")

    st.subheader("ベクトル化")
    eps = st.slider("簡略化 (Douglas-Peucker, px)", 0.5, 5.0, 1.2, 0.1)
    min_len = st.slider("最小線長 (mm)", 0, 100, 12)

up = st.file_uploader("マーキング写真をアップロード (JPG/PNG)",
                      type=["jpg", "jpeg", "png"])

if up is None:
    st.info("👆 写真をアップロードしてください。歪み補正(オルソ化)済みの画像を推奨します。")
    st.stop()

file_bytes = np.frombuffer(up.read(), np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img is None:
    st.error("画像を読み込めませんでした。")
    st.stop()

H, W = img.shape[:2]
st.write(f"画像サイズ: **{W} × {H} px** → 実寸 **{real_w:.0f} × {real_h:.0f} mm** "
         f"(解像度 {real_w / W:.2f} mm/px)")
if abs((real_w / W) / (real_h / H) - 1) > 0.05:
    st.warning("⚠️ 画像の縦横比と実寸法の縦横比が5%以上ずれています。")

with st.spinner("抽出・ベクトル化中..."):
    masks = {}
    chroma_total = np.zeros((H, W), np.uint8)   # 黒・白処理時の除外用

    red = extract_red(img, r_smin, r_vmin, r_hue, r_minarea) if use_red else None
    if red is not None:
        chroma_total |= red
    green = (extract_hue_band(img, g_hlo, g_hhi, g_smin, g_vmin, g_minarea)
             if use_green else None)
    if green is not None:
        chroma_total |= green
    blue = (extract_hue_band(img, b_hlo, b_hhi, bl_smin, bl_vmin, bl_minarea)
            if use_blue else None)
    if blue is not None:
        chroma_total |= blue

    black = (extract_black(img, chroma_total, k_vmax, k_bh, k_width, k_span,
                           rule_h, rule_v) if use_black else None)
    chalk, chalk_rej = ((extract_chalk(img, chroma_total, c_vmin, c_th, c_smax,
                                       c_wmin, c_wmax, c_cv, c_span))
                        if use_chalk else (None, None))

    # カラーチョーク抽出
    cc_masks = {}
    cc_rejs = {}
    for ck, p in cc_params.items():
        m, rej = extract_color_chalk(img, chroma_total, p["hue"], p["s_min"],
                                     p["v_min"], p["wmin"], p["wmax"],
                                     p["cvmax"], p["span"])
        cc_masks[ck] = m
        cc_rejs[ck] = rej

    layer_masks = {
        "CRACK_RED": (red, 1),
        "CRACK_GREEN": (green, 3),
        "CRACK_BLUE": (blue, 5),
        "CRACK_BLACK": (black, 7),
        "CHALK_WHITE": (chalk, 9),
    }
    for ck in cc_params:
        d = COLOR_CHALK_DEFS[ck]
        layer_masks[f"CHALK_{ck.upper()}"] = (cc_masks[ck], d["dxf"])
    dxf_bytes, dxf_doc, stats, px_polys = build_dxf(
        layer_masks, img.shape, real_w, real_h, eps, min_len,
        embed_photo=embed_photo)
    photo_png = cv2.imencode(".png", img)[1].tobytes() if embed_photo else None
    dwg_bytes, dwg_err = (export_dwg_bytes(dxf_doc, oda_path)
                          if make_dwg else (None, None))

# ---- 結果表示 ----
active = [(name, jp) for name, jp, used in [
    ("CRACK_RED", "赤", use_red), ("CRACK_GREEN", "緑", use_green),
    ("CRACK_BLUE", "青", use_blue), ("CRACK_BLACK", "黒", use_black),
    ("CHALK_WHITE", "白チョーク", use_chalk)] if used]
for ck in cc_params:
    d = COLOR_CHALK_DEFS[ck]
    active.append((f"CHALK_{ck.upper()}", f"{d['jp']}チョーク"))

cols = st.columns(len(active))
for c, (name, jp) in zip(cols, active):
    c.metric(jp, f"{stats[name][0]} 本", f"{stats[name][1] / 1000:.2f} m")

# 変換成功（ダウンロード表示到達）を1回の実行としてGA4に記録
track_event("dxf_conversion")

dl_cols = st.columns(3)
if embed_photo:
    dl_cols[0].download_button(
        "⬇️ DXF + 写真 (ZIP)",
        make_cad_zip(dxf_bytes, "crack_map.dxf", photo_png),
        file_name="crack_map_dxf.zip", mime="application/zip",
        use_container_width=True)
else:
    dl_cols[0].download_button(
        "⬇️ DXF", dxf_bytes, file_name="crack_map.dxf",
        mime="application/dxf", use_container_width=True)

if make_dwg:
    if dwg_bytes is not None:
        if embed_photo:
            dl_cols[1].download_button(
                "⬇️ DWG + 写真 (ZIP)",
                make_cad_zip(dwg_bytes, "crack_map.dwg", photo_png),
                file_name="crack_map_dwg.zip", mime="application/zip",
                use_container_width=True)
        else:
            dl_cols[1].download_button(
                "⬇️ DWG", dwg_bytes, file_name="crack_map.dwg",
                mime="application/octet-stream", use_container_width=True)
    else:
        st.error(dwg_err)

if embed_photo:
    st.info("📷 写真は crack_photo.png として外部参照されます。"
            "ZIPを展開し、CADファイルと crack_photo.png を**同じフォルダ**に"
            "置いたまま開いてください(別フォルダに移すと写真が表示されません)。")

tab1, tab2, tab3 = st.tabs(["✅ ベクトル重ね合わせ", "🎨 抽出マスク", "📐 DXFプレビュー"])

DRAW_BGR = {"CRACK_RED": (0, 255, 255), "CRACK_GREEN": (255, 0, 255),
            "CRACK_BLUE": (0, 165, 255), "CRACK_BLACK": (255, 255, 0),
            "CHALK_WHITE": (0, 255, 0)}
for ck in cc_params:
    DRAW_BGR[f"CHALK_{ck.upper()}"] = COLOR_CHALK_DEFS[ck]["draw"]
CAPTION = ("黄=赤 / マゼンタ=緑 / オレンジ=青 / 水色=黒 / 緑=白チョーク "
           "(視認性のため元の色と変えています)")

with tab1:
    overlay = img.copy()
    for name, _ in active:
        for pl in px_polys[name]:
            cv2.polylines(overlay, [pl], False, DRAW_BGR[name], 2)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=CAPTION,
             use_container_width=True)

with tab2:
    mask_map = {"CRACK_RED": red, "CRACK_GREEN": green, "CRACK_BLUE": blue,
                "CRACK_BLACK": black, "CHALK_WHITE": chalk}
    for ck in cc_params:
        mask_map[f"CHALK_{ck.upper()}"] = cc_masks[ck]
    cols2 = st.columns(2)
    idx = 0
    for name, jp in active:
        m = mask_map[name]
        if m is None:
            continue
        vis = img.copy()
        vis[m > 0] = (0, 255, 0)
        if name == "CHALK_WHITE" and chalk_rej is not None and show_rejected:
            vis[chalk_rej > 0] = (0, 0, 255)
        ck_name = name.replace("CHALK_", "").lower()
        if ck_name in cc_rejs and show_rejected:
            vis[cc_rejs[ck_name] > 0] = (0, 0, 255)
        cap = f"{jp} 抽出 (緑)"
        if (name == "CHALK_WHITE" or ck_name in cc_rejs) and show_rejected:
            cap += " / 赤=線幅不規則で棄却"
        cols2[idx % 2].image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                             caption=cap, use_container_width=True)
        idx += 1

with tab3:
    canvas = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(canvas, (2, 2), (W - 3, H - 3), (160, 160, 160), 1)
    DXF_BGR = {"CRACK_RED": (0, 0, 255), "CRACK_GREEN": (0, 160, 0),
               "CRACK_BLUE": (255, 0, 0), "CRACK_BLACK": (0, 0, 0),
               "CHALK_WHITE": (140, 140, 140)}
    for ck in cc_params:
        DXF_BGR[f"CHALK_{ck.upper()}"] = COLOR_CHALK_DEFS[ck]["draw"]
    for name, _ in active:
        for pl in px_polys[name]:
            cv2.polylines(canvas, [pl], False, DXF_BGR[name], 2)
    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
             caption="DXF出力イメージ", use_container_width=True)

st.caption("レイヤ: OUTLINE / CRACK_RED(1) / CRACK_GREEN(3) / CRACK_BLUE(5) / "
           "CRACK_BLACK(7) / CHALK_WHITE(9) ・単位mm・原点左下")
st.caption("⚠️ 白チョークは『線幅が一定』という特徴でエフロレッセンスと識別していますが、"
           "ひび割れに沿って筋状に析出したエフロはチョーク線と区別できない場合があります。"
           "必ず重ね合わせ表示で目視確認してください。")
