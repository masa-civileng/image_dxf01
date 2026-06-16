"""
Slab App — 画像結合 + ゆがみ補正 統合アプリ

手順1：画像ファイル結合（Slab Stitcher / photomerge Reposition 相当）
手順2：ゆがみ補正（ひし形・台形 → 長方形 / ホモグラフィ変換）

手順1で結合した画像は、ワンクリックで手順2へ引き継げます。

起動: streamlit run slab_app.py
依存: pip install streamlit opencv-python-headless numpy pillow streamlit-image-coordinates
"""

import io
import time
import itertools
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Slab App｜結合＆ゆがみ補正", layout="wide")
Image.MAX_IMAGE_PIXELS = None  # 高解像度の結合結果に対応


# ============================================================
# 共通ユーティリティ
# ============================================================
def read_image(uploaded) -> np.ndarray:
    """アップロードファイル -> BGR ndarray"""
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def to_jpeg_bytes(bgr: np.ndarray, quality: int = 95) -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def show_bgr(bgr: np.ndarray, caption: str = "", width=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=caption, use_container_width=(width is None), width=width)


# ============================================================
# 手順1：剛体変換ベースの結合（コアロジック）
# ============================================================
def _detect_features(images, n_features=3000):
    sift = cv2.SIFT_create(n_features)
    kps, descs = [], []
    for im in images:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k, d = sift.detectAndCompute(gray, None)
        kps.append(k)
        descs.append(d)
    return kps, descs


def _match_counts(descs, ratio=0.75):
    n = len(descs)
    bf = cv2.BFMatcher()
    mc = np.zeros((n, n), int)
    for i, j in itertools.combinations(range(n), 2):
        if descs[i] is None or descs[j] is None:
            continue
        raw = bf.knnMatch(descs[i], descs[j], k=2)
        good = 0
        for pair in raw:
            if len(pair) < 2:
                continue
            m, nn = pair
            if m.distance < ratio * nn.distance:
                good += 1
        mc[i, j] = mc[j, i] = good
    return mc


def _pair_rigid(kps, descs, i, j, ratio=0.75, min_good=15):
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(descs[i], descs[j], k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, nn = pair
        if m.distance < ratio * nn.distance:
            good.append(m)
    if len(good) < min_good:
        return None, 0
    pi = np.float32([kps[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pj = np.float32([kps[j][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(
        pi, pj, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None, 0
    return M, int(inl.sum()) if inl is not None else 0


def _build_global_transforms(kps, descs, mc, log=None):
    if log is None:
        log = []
    n = len(kps)
    ref = int(np.argmax(mc.sum(axis=1)))
    log.append(f"基準画像: #{ref + 1}")

    H = {ref: np.eye(3)}
    visited = {ref}
    edges = []
    for i, j in itertools.combinations(range(n), 2):
        edges.append((mc[i, j], i, j))
    edges.sort(reverse=True)

    while len(visited) < n:
        progressed = False
        for w, i, j in edges:
            if (i in visited) ^ (j in visited):
                known, unknown = (i, j) if i in visited else (j, i)
                M, inl = _pair_rigid(kps, descs, unknown, known)
                if M is None:
                    continue
                Mh = np.vstack([M, [0, 0, 1]])
                H[unknown] = H[known] @ Mh
                visited.add(unknown)
                log.append(f"#{unknown + 1} を #{known + 1} に接続 "
                           f"(対応点 {inl}, マッチ {w})")
                progressed = True
                break
        if not progressed:
            missing = sorted(set(range(n)) - visited)
            log.append(f"接続できない画像: {[m + 1 for m in missing]}")
            break

    return [H.get(i) for i in range(n)]


def _blend_canvas(images, Hmats, feather=True):
    n = len(images)
    use = [i for i in range(n) if Hmats[i] is not None]

    corners = []
    for i in use:
        h, w = images[i].shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, Hmats[i]))
    allc = np.concatenate(corners).reshape(-1, 2)
    xmin, ymin = np.floor(allc.min(0)).astype(int)
    xmax, ymax = np.ceil(allc.max(0)).astype(int)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], float)
    W, Hc = int(xmax - xmin), int(ymax - ymin)

    canvas = np.zeros((Hc, W, 3), np.float32)
    wsum = np.zeros((Hc, W), np.float32)
    for i in use:
        h, w = images[i].shape[:2]
        M = T @ Hmats[i]
        warped = cv2.warpPerspective(images[i], M, (W, Hc))
        if feather:
            yy, xx = np.mgrid[0:h, 0:w]
            wmap = np.minimum.reduce(
                [xx, w - 1 - xx, yy, h - 1 - yy]).astype(np.float32)
            wmap /= max(wmap.max(), 1e-6)
        else:
            wmap = np.ones((h, w), np.float32)
        wmap_w = cv2.warpPerspective(wmap, M, (W, Hc))
        canvas += warped.astype(np.float32) * wmap_w[..., None]
        wsum += wmap_w
        del warped, wmap_w

    out = np.zeros_like(canvas)
    nz = wsum > 1e-6
    out[nz] = canvas[nz] / wsum[nz, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def stitch_rigid(images, work_scale=0.35, output_scale=0.7, feather=True, log=None):
    if log is None:
        log = []
    t0 = time.time()

    work = [cv2.resize(im, None, fx=work_scale, fy=work_scale) for im in images]
    kps, descs = _detect_features(work)
    mc = _match_counts(descs)
    Hwork = _build_global_transforms(kps, descs, mc, log=log)

    r = output_scale / work_scale
    S = np.diag([r, r, 1.0])
    Sinv = np.diag([1 / r, 1 / r, 1.0])
    Hout = [None if Hw is None else S @ Hw @ Sinv for Hw in Hwork]

    out_imgs = [cv2.resize(im, None, fx=output_scale, fy=output_scale)
                for im in images]
    result = _blend_canvas(out_imgs, Hout, feather=feather)
    log.append(f"合成サイズ: {result.shape[1]}×{result.shape[0]}px "
               f"／ 所要 {time.time() - t0:.1f}s")
    return result


def crop_bbox(pano):
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pano
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return pano[y:y + h, x:x + w]


def crop_inner(pano, fill_ratio=0.85):
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8)
    col = mask.mean(axis=0)
    row = mask.mean(axis=1)
    cs = np.where(col >= fill_ratio)[0]
    rs = np.where(row >= fill_ratio)[0]
    if len(cs) == 0 or len(rs) == 0:
        return crop_bbox(pano)
    return pano[rs[0]:rs[-1] + 1, cs[0]:cs[-1] + 1]


# ============================================================
# セッション初期化
# ============================================================
def init_state():
    ss = st.session_state
    ss.setdefault("stitched_bgr", None)     # 手順1の結合結果 (BGR ndarray)
    ss.setdefault("corr_source_bgr", None)  # 手順2の入力 (BGR ndarray)
    ss.setdefault("corr_source_name", None) # 入力の由来表示用
    ss.setdefault("pts", [])                # 手順2 四隅(表示座標)
    ss.setdefault("last_click", None)
    ss.setdefault("corr_img_id", None)      # 手順2 入力識別子


init_state()


# ============================================================
# 手順1 タブ
# ============================================================
def render_stitch_tab():
    st.header("手順1：画像ファイル結合")
    st.caption("真上から分割撮影した平面画像を、外縁を歪めずに1枚へ結合します"
               "（Photoshop photomerge の Reposition 相当）。")

    with st.expander("結合の設定", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            output_scale = st.slider(
                "出力解像度", 0.3, 1.0, 0.7, 0.1,
                help="大きいほど高精細。処理時間とメモリも増えます。まず0.7で試行。")
            work_scale = st.slider(
                "解析解像度", 0.2, 0.6, 0.35, 0.05,
                help="位置合わせ精度。縁が合わないときは上げる（遅くなる）。")
        with c2:
            feather = st.checkbox(
                "継ぎ目をなじませる", value=True,
                help="重なり部分を距離重みでブレンドし継ぎ目を目立たなくします。")
            crop_mode = st.radio(
                "トリミング",
                ["被写体全体（外接）", "長方形に近づける（内接）", "なし"],
                help="外接=被写体が必ず収まる。内接=四隅の黒欠けを抑える。")

    files = st.file_uploader(
        "分割画像をアップロード（2枚以上、撮影順でなくてOK）",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True, key="stitch_uploader")

    if not files:
        st.info("画像をアップロードすると、ここにプレビューが表示されます。")
        st.markdown(
            "**使い方**\n\n"
            "1. 真上から少しずつ重ねて分割撮影した画像をまとめて選択\n"
            "2. 設定（まずは初期値のまま）を確認\n"
            "3. 「結合する」を押す\n\n"
            "向き（回転・上下逆）がばらばらでも、重なりが3〜5割あれば自動でつながります。")
    else:
        imgs = [read_image(f) for f in files]
        imgs = [im for im in imgs if im is not None]
        st.write(f"読み込み: {len(imgs)} 枚")

        thumbs = st.columns(min(len(imgs), 6))
        for i, im in enumerate(imgs):
            with thumbs[i % len(thumbs)]:
                show_bgr(im, f"#{i + 1}")

        if st.button("結合する", type="primary", disabled=(len(imgs) < 2)):
            if len(imgs) < 2:
                st.error("2枚以上の画像が必要です。")
            else:
                log = []
                with st.spinner("位置合わせして結合しています…"):
                    try:
                        pano = stitch_rigid(
                            imgs, work_scale=work_scale,
                            output_scale=output_scale, feather=feather, log=log)
                        if crop_mode.startswith("被写体全体"):
                            pano = crop_bbox(pano)
                        elif crop_mode.startswith("長方形"):
                            pano = crop_inner(pano)
                    except Exception as e:
                        st.error(f"結合に失敗しました: {e}")
                        pano = None

                with st.expander("処理ログ", expanded=False):
                    for line in log:
                        st.text(line)

                if pano is not None:
                    st.session_state.stitched_bgr = pano

    # 結合結果（前回分も保持表示）
    pano = st.session_state.stitched_bgr
    if pano is not None:
        st.divider()
        st.success(f"結合結果: {pano.shape[1]}×{pano.shape[0]}px")
        show_bgr(pano, "結合結果")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "結合画像をダウンロード（JPEG）",
                data=to_jpeg_bytes(pano),
                file_name="stitched.jpg",
                mime="image/jpeg", use_container_width=True)
        with c2:
            if st.button("この画像を手順2へ送る →", type="primary",
                         use_container_width=True):
                st.session_state.corr_source_bgr = pano.copy()
                st.session_state.corr_source_name = "手順1の結合結果"
                # 入力が変わったので四隅をリセット
                st.session_state.pts = []
                st.session_state.last_click = None
                st.session_state.corr_img_id = f"stitched_{id(pano)}"
                st.success("手順2へ送りました。上の「手順2：ゆがみ補正」タブを開いてください。")


# ============================================================
# 手順2 タブ
# ============================================================
def render_correct_tab():
    st.header("手順2：ゆがみ補正")
    st.caption("ひし形・台形に写った被写体を長方形に補正します。"
               "被写体の四隅を「① 左上 → ② 右上 → ③ 右下 → ④ 左下」の順にクリックしてください。")

    # --- 入力ソースの決定（手順1から or 直接アップロード） ---
    with st.expander("入力画像", expanded=True):
        if st.session_state.corr_source_bgr is not None:
            st.info(f"現在の入力: {st.session_state.corr_source_name}")
        up = st.file_uploader("または画像を直接アップロード",
                              type=["jpg", "jpeg", "png"], key="corr_uploader")
        if up is not None:
            new_id = f"upload_{up.file_id}"
            if st.session_state.corr_img_id != new_id:
                st.session_state.corr_source_bgr = read_image(up)
                st.session_state.corr_source_name = f"アップロード: {up.name}"
                st.session_state.corr_img_id = new_id
                st.session_state.pts = []
                st.session_state.last_click = None

    if st.session_state.corr_source_bgr is None:
        st.warning("入力画像がありません。手順1で「手順2へ送る」を押すか、上で画像をアップロードしてください。")
        return

    src_bgr = st.session_state.corr_source_bgr
    pil_img = Image.fromarray(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
    orig = np.array(pil_img)
    orig_h, orig_w = orig.shape[:2]

    # --- 出力サイズ・表示設定 ---
    with st.expander("出力サイズ・表示設定", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            out_w_mm = st.number_input("横の長さ (mm)", min_value=1, value=2000, step=10)
            out_h_mm = st.number_input("縦の長さ (mm)", min_value=1, value=1500, step=10)
        with c2:
            px_per_mm = st.slider("解像度 (px/mm)", 0.2, 2.0, 1.0, 0.1,
                                  help="1.0 なら 2000mm → 2000px")
            canvas_w = st.slider("作業画面の幅 (px)", 400, 1400, 900, 50)
        with c3:
            show_guides = st.checkbox("補正後にグリッドを重ねる", value=True)
            grid_div = st.number_input("グリッド分割数", min_value=2, max_value=50, value=10)

    out_w_px = int(round(out_w_mm * px_per_mm))
    out_h_px = int(round(out_h_mm * px_per_mm))
    st.caption(f"出力画像: {out_w_px} × {out_h_px} px")

    scale = canvas_w / orig_w
    canvas_h = int(round(orig_h * scale))
    disp_base = pil_img.resize((canvas_w, canvas_h))

    st.subheader("四隅を指定")
    st.caption("順番：**① 左上 → ② 右上 → ③ 右下 → ④ 左下**。やり直すときは下のボタンを使ってください。")

    col_canvas, col_preview = st.columns([1, 1])

    with col_canvas:
        b1, b2 = st.columns(2)
        if b1.button("最後の点を取消", use_container_width=True):
            if st.session_state.pts:
                st.session_state.pts.pop()
        if b2.button("全リセット", use_container_width=True):
            st.session_state.pts = []
            st.session_state.last_click = None

        disp = disp_base.copy()
        draw = ImageDraw.Draw(disp)
        labels = ["1", "2", "3", "4"]
        for i, (x, y) in enumerate(st.session_state.pts):
            r = 7
            draw.ellipse([x - r, y - r, x + r, y + r],
                         outline="#ffffff", width=2, fill="#ff3030")
            draw.text((x + 9, y - 6), labels[i], fill="#ffff00")
        if len(st.session_state.pts) == 4:
            draw.line(st.session_state.pts + [st.session_state.pts[0]],
                      fill="#30ff60", width=2)

        if len(st.session_state.pts) < 4:
            coords = streamlit_image_coordinates(disp, key="corr_picker")
            if coords is not None:
                click = (float(coords["x"]), float(coords["y"]))
                if click != st.session_state.last_click:
                    st.session_state.last_click = click
                    st.session_state.pts.append(click)
                    st.rerun()
        else:
            st.image(disp, use_container_width=True)

        names = ["① 左上", "② 右上", "③ 右下", "④ 左下"]
        msg = [f"✅ {nm}" if i < len(st.session_state.pts) else f"⬜ {nm}"
               for i, nm in enumerate(names)]
        st.write("　".join(msg))

    with col_preview:
        st.markdown("**補正結果プレビュー**")
        pts = st.session_state.pts
        if len(pts) < 4:
            st.info(f"あと {4 - len(pts)} 点クリックしてください。")
        else:
            src = np.array([[x / scale, y / scale] for (x, y) in pts], dtype=np.float32)
            dst = np.array(
                [[0, 0],
                 [out_w_px - 1, 0],
                 [out_w_px - 1, out_h_px - 1],
                 [0, out_h_px - 1]],
                dtype=np.float32,
            )
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(orig, M, (out_w_px, out_h_px),
                                         flags=cv2.INTER_CUBIC)

            preview = warped.copy()
            if show_guides:
                g = (0, 200, 255)
                for i in range(1, grid_div):
                    x = int(round(out_w_px * i / grid_div))
                    cv2.line(preview, (x, 0), (x, out_h_px - 1), g, 1)
                for j in range(1, grid_div):
                    y = int(round(out_h_px * j / grid_div))
                    cv2.line(preview, (0, y), (out_w_px - 1, y), g, 1)

            st.image(preview, caption=f"{out_w_mm}×{out_h_mm}mm に補正",
                     use_container_width=True)

            out_pil = Image.fromarray(warped)
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            st.download_button(
                "補正画像をダウンロード (PNG)",
                data=buf.getvalue(),
                file_name="corrected.png",
                mime="image/png")
            st.caption(f"1 px = {1 / px_per_mm:.3f} mm")


# ============================================================
# メイン：タブ構成
# ============================================================
st.title("🧱 Slab App｜画像結合 ＆ ゆがみ補正")
st.caption("手順1で分割画像を1枚に結合し、手順2でひし形・台形を長方形に補正します。")

tab1, tab2 = st.tabs(["① 手順1：画像ファイル結合", "② 手順2：ゆがみ補正"])
with tab1:
    render_stitch_tab()
with tab2:
    render_correct_tab()
