"""hs01 发型迁移的四个 ComfyUI 节点（2026-11-08）。

设计要点（和 workflow 里的说明保持一致）：
- LoRA 只见过「头宽 ÷ 长边 0.29–0.53」的头肩 / 半身白模。直接拿头到腰的裁切换进去，头会太小（约 0.2），
  所以这里出两张裁切：① lora_white = 按头宽 ×2.4 的头肩 3:4 裁切（头宽约 31% 长边），给 hs01 LoRA；
  ② waist_white / waist_photo = 头到腰的 3:4 裁切，给 Qwen 和最终拼回。两张都从同一张整图、同一台相机裁。
- LoRA 出图后 HS01PlaceHairRef 把「带发型的头肩白模」按坐标放回腰部白模，补成 Qwen 的图2。
- Qwen 出图后 HS01Composite 只把「新头发 + 旧头发被改掉的地方」拼回原图；脸用椭圆保护，默认一个像素不动。
- 白模背景是纯黑；hs01 出图背景是灰（训练目标就是这样）。放回时用「白模剪影 + 生成图与白模的差异」做前景，
  不会把灰底带进腰部白模。

所有节点只取 batch 的第一张；尺寸用 PIL LANCZOS，mask 用 LANCZOS 再夹回 0–1；没有额外模型依赖。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

try:  # ComfyUI 自带 opencv；没有时退回 PIL
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _rgb(img: torch.Tensor) -> np.ndarray:
    """IMAGE [B,H,W,C] → 第一张 uint8 RGB [H,W,3]。"""
    a = img[0].detach().to(torch.float32).cpu().numpy()
    a = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if a.shape[2] == 4:
        a = a[..., :3]
    return np.ascontiguousarray(a)


def _mask(mask, h: int, w: int) -> np.ndarray:
    """MASK [B,H,W] / [H,W] → float32 [h,w]，0–1；尺寸不对就 LANCZOS。"""
    if mask is None:
        return np.ones((h, w), np.float32)
    a = mask
    if isinstance(a, torch.Tensor):
        if a.ndim == 3:
            a = a[0]
        a = a.detach().to(torch.float32).cpu().numpy()
    a = np.asarray(a, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != (h, w):
        a = np.asarray(Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8)).resize((w, h), Image.LANCZOS),
                       np.float32) / 255.0
    return np.clip(a, 0, 1)


def _img_t(a: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(a.astype(np.float32) / 255.0))[None]


def _mask_t(a: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(np.clip(a, 0, 1).astype(np.float32)))[None]


def _resize_rgb(a: np.ndarray, w: int, h: int) -> np.ndarray:
    w, h = max(1, int(w)), max(1, int(h))
    if a.shape[1] == w and a.shape[0] == h:
        return a
    # 大幅缩小（比如 SeedVR2 2048 → 原图 800）用 BOX / 面积平均；
    # LANCZOS 在细条纹上会产生摩尔纹和横向色带。
    if w * h * 2.25 < a.shape[0] * a.shape[1]:
        resample = Image.Resampling.BOX
    else:
        resample = Image.LANCZOS
    return np.asarray(Image.fromarray(a).resize((w, h), resample), np.uint8)


def _resize_mask(a: np.ndarray, w: int, h: int) -> np.ndarray:
    w, h = max(1, int(w)), max(1, int(h))
    if a.shape[1] == w and a.shape[0] == h:
        return a.astype(np.float32)
    im = Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32).mean(2) / 255.0


def _dilate(a: np.ndarray, px: int) -> np.ndarray:
    px = max(0, int(px))
    if px <= 0:
        return a.astype(bool)
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        return cv2.dilate(a.astype(np.uint8), k, iterations=1).astype(bool)
    im = Image.fromarray((a.astype(np.uint8) * 255))
    return np.asarray(im.filter(ImageFilter.MaxFilter(2 * px + 1)), np.float32) > 127


def _blur(a: np.ndarray, px: float) -> np.ndarray:
    if px <= 0:
        return np.clip(a, 0, 1)
    if cv2 is not None:
        k = int(round(px * 2 + 1)) | 1
        return np.clip(cv2.GaussianBlur(a.astype(np.float32), (k, k), px), 0, 1)
    im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(px)), np.float32) / 255.0


def _close(a: np.ndarray, px: int) -> np.ndarray:
    px = max(0, int(px))
    if px <= 0 or cv2 is None:
        return a.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)


def _largest_components(mask: np.ndarray, min_frac: float = 0.0002) -> np.ndarray:
    """去掉小碎块（默认小于画面 0.02%）。"""
    if cv2 is None:
        return mask.astype(bool)
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return mask.astype(bool)
    min_area = max(4, int(mask.size * min_frac))
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep.astype(bool)


def _resolve(path: str) -> Path:
    s = (path or "").strip()
    p = Path(s).expanduser()
    if p.is_absolute():
        return p
    try:
        import folder_paths
        return Path(folder_paths.get_output_directory()) / p
    except Exception:  # noqa: BLE001
        return p


def _read_json(path: str) -> dict:
    p = _resolve(path)
    if p.is_dir():
        p = p / "render.json"
    return json.loads(p.read_text())


def _keypoints(render_json: str) -> tuple[np.ndarray | None, dict]:
    """从 render.json 取整张照片像素坐标的 70 个关键点；优先标定体型（调姿势后）。"""
    rep = _read_json(render_json)
    rep = rep.get("report", rep)
    for sec in ("calibrated", "perspective", ""):
        d = rep.get(sec, rep) if sec else rep
        if not isinstance(d, dict):
            continue
        for key in ("keypoints_px", "sanshi_keypoints_px"):
            kp = d.get(key)
            if kp is not None:
                try:
                    a = np.asarray(kp, np.float64)
                except Exception:  # noqa: BLE001
                    continue
                if a.ndim == 2 and a.shape[0] >= 11 and a.shape[1] >= 2:
                    return a[:, :2], rep
    return None, rep


def _clamp_box(x0: float, y0: float, x1: float, y1: float, W: int, H: int) -> tuple[int, int, int, int]:
    w = max(16.0, min(float(W), x1 - x0))
    h = max(16.0, min(float(H), y1 - y0))
    x0 = min(max(0.0, x0), W - w)
    y0 = min(max(0.0, y0), H - h)
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x0 + w)), int(round(y0 + h))
    return x0, y0, min(x1, W), min(y1, H)


def _face_geometry(kp: np.ndarray | None, head_w: float, face_cx: float, face_cy: float) -> tuple[float, float, float, float]:
    """脸中心、脸宽、脸高（用于椭圆保护）。优先两耳 / 两眼距离。"""
    if kp is not None and len(kp) >= 5:
        eye = float(np.linalg.norm(kp[1] - kp[2]))
        ear = float(np.linalg.norm(kp[3] - kp[4]))
        fw = max(ear * 1.05, eye * 2.1, head_w * 0.55)
    else:
        fw = head_w * 0.72
    fh = fw * 1.35
    return float(face_cx), float(face_cy) + 0.02 * fh, fw, fh


def _ellipse_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float, blur: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    m = (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0).astype(np.float32)
    if blur > 0:
        m = _blur(m, blur)
    return np.clip(m, 0, 1)


# ---------------------------------------------------------------------------
# 1. 裁切
# ---------------------------------------------------------------------------

def _lora_bucket(aspect: float) -> tuple[int, int]:
    """LoRA 训练只接受 3:4 = 1152×1536、4:5 = 1152×1440；按宽高比选最近的一档。"""
    buckets = {0.75: (1152, 1536), 0.8: (1152, 1440)}
    return min(buckets.values(), key=lambda wh: abs(math.log((wh[0] / wh[1]) / max(1e-6, aspect))))


class HS01OneCrop:
    CATEGORY = "hs01"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("lora_white", "lora_mask", "waist_white", "waist_photo", "waist_photo_native",
                    "waist_face_mask", "person_mask", "info", "report", "work_width", "work_height")
    OUTPUT_TOOLTIPS = (
        "头肩白模（从同一个手动框内部派生，给 hs01）",
        "头肩白模剪影",
        "头到腰白模（手动框，1152×1536 档）",
        "头到腰照片（手动框，1152×1536 档）",
        "头到腰照片（原像素，拼回用）",
        "脸保护椭圆（兼容旧节点，工作尺寸）",
        "人物遮罩（兼容旧节点，工作尺寸）",
        "裁切框 / 工作尺寸 / 坐标（JSON）",
        "中文摘要",
        "工作宽度",
        "工作高度",
    )
    DESCRIPTION = ("只需要手动裁一次：用户只给一个头到腰框（含手臂），"
                   "头肩框固定取这个框内部 77.5% 宽、顶对齐的子框；不做第二个裁切框，也不做头肩检测。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "white_image": ("IMAGE", {"tooltip": "ScenePose 白模图"}),
            "white_mask": ("MASK", {"tooltip": "ScenePose 白模遮罩"}),
            "photo_image": ("IMAGE", {"tooltip": "整张原照片"}),
            "render_json": ("STRING", {"forceInput": True, "tooltip": "ScenePose render_json；用来派生头肩框和脸保护"}),
            "manual_waist_x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1, "tooltip": "唯一手动框 x（整张照片像素）"}),
            "manual_waist_y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1, "tooltip": "唯一手动框 y"}),
            "manual_waist_w": ("INT", {"default": 1, "min": 1, "max": 16384, "step": 1, "tooltip": "唯一手动框宽"}),
            "manual_waist_h": ("INT", {"default": 1, "min": 1, "max": 16384, "step": 1, "tooltip": "唯一手动框高"}),
        }}

    def run(self, white_image, white_mask, photo_image, render_json,
            manual_waist_x, manual_waist_y, manual_waist_w, manual_waist_h):
        white = _rgb(white_image)
        photo = _rgb(photo_image)
        H, W = white.shape[:2]
        if photo.shape[:2] != (H, W):
            raise ValueError(f"白模 {W}×{H} 和原图 {photo.shape[1]}×{photo.shape[0]} 尺寸不同；要接同一张照片的 ScenePose 输出")
        wm = _mask(white_mask, H, W)
        if float(wm.max()) < 0.01:
            wm = (_gray(white) > 0.03).astype(np.float32)
        wm_b = wm > 0.5

        wx0, wy0, wx1, wy1 = _clamp_box(float(manual_waist_x), float(manual_waist_y),
                                        float(manual_waist_x + manual_waist_w), float(manual_waist_y + manual_waist_h), W, H)
        waist_w, waist_h = wx1 - wx0, wy1 - wy0
        if waist_w < 16 or waist_h < 16:
            raise ValueError(f"手动框太小：{waist_w}×{waist_h}")

        # 唯一手动框 = 头到腰（含手臂）。头肩框只在它内部派生，用户不用再裁第二次。
        kp, _ = _keypoints(render_json)
        if kp is not None and len(kp) >= 5 and np.isfinite(kp[:5]).all():
            face_cx, face_cy = float(kp[0, 0]), float(kp[0, 1])
            ear = float(np.linalg.norm(kp[3] - kp[4]))
            head_w = max(22.0, ear * 3.0, 0.26 * waist_w)
        else:
            face_cx, face_cy = wx0 + 0.50 * waist_w, wy0 + 0.38 * waist_h
            head_w = max(22.0, 0.30 * waist_w)

        # 头肩子框不靠检测：固定取手动框内 77.5% 宽、顶对齐、水平居中。
        # 这样用户只裁一次，头肩景别仍然稳定（头宽约 31% 长边）。
        lora_w = 0.775 * waist_w
        lora_h = lora_w / 0.75
        if lora_h > waist_h:
            lora_h = waist_h
            lora_w = lora_h * 0.75
        lora_w, lora_h = max(16.0, lora_w), max(16.0, lora_h)
        lx0 = wx0 + (waist_w - lora_w) / 2
        ly0 = wy0
        lx1, ly1 = lx0 + lora_w, ly0 + lora_h
        lx0, ly0, lx1, ly1 = _clamp_box(lx0, ly0, lx1, ly1, W, H)
        lora_w, lora_h = lx1 - lx0, ly1 - ly0
        if not (wx0 <= lx0 and wy0 <= ly0 and wx1 >= lx1 and wy1 >= ly1):
            raise ValueError(f"派生头肩框 [{lx0},{ly0},{lx1},{ly1}] 超出唯一手动框 [{wx0},{wy0},{wx1},{wy1}]；把手动框放宽一点")

        ratio = waist_w / max(1.0, waist_h)
        work_w, work_h = (1152, 1440) if abs(math.log(ratio / 0.8)) < abs(math.log(ratio / 0.75)) else (1152, 1536)
        sx, sy = work_w / float(waist_w), work_h / float(waist_h)

        white_lora = _resize_rgb(white[ly0:ly1, lx0:lx1], work_w, work_h)
        mask_lora = _resize_mask(wm_b[ly0:ly1, lx0:lx1].astype(np.float32), work_w, work_h)
        white_waist = _resize_rgb(white[wy0:wy1, wx0:wx1], work_w, work_h)
        photo_waist = _resize_rgb(photo[wy0:wy1, wx0:wx1], work_w, work_h)
        photo_native = photo[wy0:wy1, wx0:wx1].copy()

        if kp is not None and len(kp) >= 5 and np.isfinite(kp[:5]).all():
            face_cx, face_cy, fw, fh = _face_geometry(kp, head_w, face_cx, face_cy)
        else:
            fw, fh = 0.42 * lora_w, 0.45 * lora_h
        ex, ey = face_cx - wx0, face_cy - wy0
        face_native = _ellipse_mask(waist_h, waist_w, ex, ey, 0.55 * fw, 0.62 * fh, max(2.0, 0.02 * fw))
        face_work = _resize_mask(face_native, work_w, work_h)
        person_work = _resize_mask(wm_b[wy0:wy1, wx0:wx1].astype(np.float32), work_w, work_h)

        info = {
            "full_size": [W, H],
            "lora_box": [lx0, ly0, lx1, ly1],
            "waist_box": [wx0, wy0, wx1, wy1],
            "lora_size": [lora_w, lora_h],
            "lora_work_size": [work_w, work_h],
            "waist_size": [waist_w, waist_h],
            "work_size": [work_w, work_h],
            "work_scale": [sx, sy],
            "head_width": head_w,
            "head_top": ly0 + 0.05 * lora_h,
            "face_center": [face_cx, face_cy],
            "face_size": [fw, fh],
            "waist_y": wy1,
            "shoulder_y": ly1,
            "chest_y": wy1,
            "head_ratio_lora": head_w / max(lora_w, lora_h),
            "manual": "one_crop_waist; derived lora",
            "source": "hs01_hair_transfer/one_crop/1",
        }
        report = "\n".join([
            f"整张照片 {W}×{H}（只手动裁一次：头到腰）",
            f"手动框（Qwen/成片）：[{wx0},{wy0},{wx1},{wy1}] {waist_w}×{waist_h}",
            f"内部派生头肩（hs01）：[{lx0},{ly0},{lx1},{ly1}] {lora_w}×{lora_h}（手动框内 77.5% 内缩）",
            f"工作尺寸：{work_w}×{work_h}（缩放 {sx:.3f}×{sy:.3f}）",
            f"头宽估计 {head_w:.0f}px = 头肩长边 {head_w / max(lora_w, lora_h) * 100:.1f}%",
            "只裁这一次；头肩框是内部派生，不需要第二个裁切框。",
        ])
        return ( _img_t(white_lora), _mask_t(mask_lora), _img_t(white_waist), _img_t(photo_waist),
                 _img_t(photo_native), _mask_t(face_work), _mask_t(person_work),
                 json.dumps(info, ensure_ascii=False), report, work_w, work_h )


class HS01HairMaskToWaist:
    CATEGORY = "hs01"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("hair_mask", "report")
    DESCRIPTION = ("把 hs01 出图里的头发抠成遮罩，按坐标映射到腰部画布；不生成第二张白模图。"
                   "Qwen 的图2 直接接 hs01 出图，这个节点只给最终拼回提供头发遮罩。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "generated_lora": ("IMAGE", {"tooltip": "hs01 还原到头肩裁切尺寸的出图"}),
            "lora_white": ("IMAGE", {"tooltip": "HS01HeadWaistCrop 的 lora_white（光头白模）"}),
            "lora_mask": ("MASK", {"tooltip": "HS01HeadWaistCrop 的 lora_mask"}),
            "waist_white": ("IMAGE", {"tooltip": "HS01HeadWaistCrop 的 waist_white（只用来拿工作尺寸）"}),
            "info": ("STRING", {"forceInput": True, "tooltip": "HS01HeadWaistCrop 的 info"}),
            "feather": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1, "tooltip": "头发遮罩羽化像素"}),
            "grow": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.8, "step": 0.02,
                               "tooltip": "头发允许超出光头剪影多少（头宽的比例）"}),
        }}

    def run(self, generated_lora, lora_white, lora_mask, waist_white, info, feather, grow):
        inf = json.loads(info)
        gen = _rgb(generated_lora)
        bald = _rgb(lora_white)
        H, W = bald.shape[:2]
        if gen.shape[:2] != (H, W):
            gen = _resize_rgb(gen, W, H)
        sil = _mask(lora_mask, H, W) > 0.5
        if float(sil.mean()) < 0.001:
            sil = _gray(bald) > 0.03
        head_w = float(inf.get("head_width", W * 0.3))
        grow_px = max(2, int(round(float(grow) * head_w)))
        near = _dilate(sil, grow_px)

        dark = _gray(gen) < 0.22
        top_region = np.zeros((H, W), bool)
        ylim = int(np.clip((inf.get("waist_y", H) - inf["lora_box"][1]) * (H / max(1.0, inf["lora_size"][1])), 0, H))
        top_region[:max(1, ylim), :] = True
        fcx = (inf["face_center"][0] - inf["lora_box"][0]) * (W / max(1.0, inf["lora_size"][0]))
        fcy = (inf["face_center"][1] - inf["lora_box"][1]) * (H / max(1.0, inf["lora_size"][1]))
        fw = inf["face_size"][0] * (W / max(1.0, inf["lora_size"][0]))
        fh = inf["face_size"][1] * (H / max(1.0, inf["lora_size"][1]))
        face = _ellipse_mask(H, W, fcx, fcy + 0.16 * fh, 0.42 * fw, 0.30 * fh, 2.0) > 0.3
        hair = dark & near & top_region & (~face)
        hair = _close(hair, 2)
        hair = _largest_components(hair, 0.0008)
        if not hair.any():
            hair = dark & near & top_region
        alpha = _blur(hair.astype(np.float32), max(0.0, float(feather) / 2.0))

        work_w = int(waist_white.shape[2])
        work_h = int(waist_white.shape[1])
        sx, sy = inf["work_scale"]
        lx, ly = inf["lora_box"][:2]
        wx, wy = inf["waist_box"][:2]
        tx = int(round((lx - wx) * sx))
        ty = int(round((ly - wy) * sy))
        tw = max(1, int(round((inf["lora_box"][2] - inf["lora_box"][0]) * sx)))
        th = max(1, int(round((inf["lora_box"][3] - inf["lora_box"][1]) * sy)))
        sx0, sy0 = max(0, tx), max(0, ty)
        sx1, sy1 = min(work_w, tx + tw), min(work_h, ty + th)
        if sx1 <= sx0 or sy1 <= sy0:
            raise ValueError("头肩框不在腰部画布里；检查 crop 输出")
        alpha_r = _resize_mask(alpha, tw, th)
        ref_mask = np.zeros((work_h, work_w), np.float32)
        ref_mask[sy0:sy1, sx0:sx1] = alpha_r[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx]
        report = "\n".join([
            f"头发遮罩：头肩 {gen.shape[1]}×{gen.shape[0]} → 腰部 {work_w}×{work_h} 的 [{sx0},{sy0},{sx1},{sy1}]",
            f"只抠头发：暗部 {int(dark.sum())} px，扣掉脸后 {int(hair.sum())} px；羽化 {feather}px",
        ])
        return _mask_t(ref_mask), report


class HS01PhotoHairRef:
    CATEGORY = "hs01"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("hair_ref", "hair_mask", "report")
    DESCRIPTION = ("把 hs01 出图里的头发按坐标贴到「原图腰部裁切」上，做成 Qwen 的图2："
                   "和原图同一个人、同一构图、含手臂和正常背景，只多了目标发型。"
                   "hs01 仍然只在头肩裁切上生成一次，这不是第二次生成。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "generated_lora": ("IMAGE", {"tooltip": "hs01 出图（1152×1536，头肩裁切）"}),
            "lora_white": ("IMAGE", {"tooltip": "HS01HeadWaistCrop 的 lora_white"}),
            "lora_mask": ("MASK", {"tooltip": "HS01HeadWaistCrop 的 lora_mask"}),
            "waist_photo": ("IMAGE", {"tooltip": "HS01HeadWaistCrop 的 waist_photo（1152×1536，含手臂）"}),
            "info": ("STRING", {"forceInput": True, "tooltip": "HS01HeadWaistCrop 的 info"}),
            "feather": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1, "tooltip": "头发边缘羽化像素"}),
            "grow": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.8, "step": 0.02,
                               "tooltip": "头发允许超出光头剪影多少（头宽的比例）"}),
        }}

    def run(self, generated_lora, lora_white, lora_mask, waist_photo, info, feather, grow):
        inf = json.loads(info)
        gen = _rgb(generated_lora)
        bald = _rgb(lora_white)
        H, W = bald.shape[:2]
        if gen.shape[:2] != (H, W):
            gen = _resize_rgb(gen, W, H)
        sil = _mask(lora_mask, H, W) > 0.5
        if float(sil.mean()) < 0.001:
            sil = _gray(bald) > 0.03
        head_w = float(inf.get("head_width", W * 0.3))
        near = _dilate(sil, max(2, int(round(float(grow) * head_w))))
        dark = _gray(gen) < 0.22
        top = np.zeros((H, W), bool)
        ylim = int(np.clip((inf.get("waist_y", H) - inf["lora_box"][1]) * (H / max(1.0, inf["lora_size"][1])), 0, H))
        top[:max(1, ylim), :] = True
        fcx = (inf["face_center"][0] - inf["lora_box"][0]) * (W / max(1.0, inf["lora_size"][0]))
        fcy = (inf["face_center"][1] - inf["lora_box"][1]) * (H / max(1.0, inf["lora_size"][1]))
        fw = inf["face_size"][0] * (W / max(1.0, inf["lora_size"][0]))
        fh = inf["face_size"][1] * (H / max(1.0, inf["lora_size"][1]))
        face = _ellipse_mask(H, W, fcx, fcy + 0.16 * fh, 0.42 * fw, 0.30 * fh, 2.0) > 0.3
        hair = dark & near & top & (~face)
        hair = _close(hair, 2)
        hair = _largest_components(hair, 0.0008)
        if not hair.any():
            hair = dark & near & top
        alpha = _blur(hair.astype(np.float32), max(0.0, float(feather) / 2.0))

        work_w = int(waist_photo.shape[2])
        work_h = int(waist_photo.shape[1])
        sx, sy = inf["work_scale"]
        lx, ly = inf["lora_box"][:2]
        wx, wy = inf["waist_box"][:2]
        tx = int(round((lx - wx) * sx))
        ty = int(round((ly - wy) * sy))
        tw = max(1, int(round((inf["lora_box"][2] - inf["lora_box"][0]) * sx)))
        th = max(1, int(round((inf["lora_box"][3] - inf["lora_box"][1]) * sy)))
        sx0, sy0 = max(0, tx), max(0, ty)
        sx1, sy1 = min(work_w, tx + tw), min(work_h, ty + th)
        gen_r = _resize_rgb(gen, tw, th)
        alpha_r = _resize_mask(alpha, tw, th)
        canvas = _rgb(waist_photo).copy()
        sub = canvas[sy0:sy1, sx0:sx1].astype(np.float32)
        gsub = gen_r[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx].astype(np.float32)
        asub = alpha_r[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx][..., None]
        canvas[sy0:sy1, sx0:sx1] = np.clip(sub * (1 - asub) + gsub * asub + 0.5, 0, 255).astype(np.uint8)
        ref_mask = np.zeros((work_h, work_w), np.float32)
        ref_mask[sy0:sy1, sx0:sx1] = asub[..., 0]
        report = "\n".join([
            f"Qwen 图2：原图腰部裁切 {work_w}×{work_h} + hs01 头发 [{sx0},{sy0},{sx1},{sy1}]",
            f"头发遮罩 {int(hair.sum())} px；含手臂 / 原背景，不生成第二张白模",
        ])
        return _img_t(canvas), _mask_t(ref_mask), report


class HS01PlaceHairRef:


    CATEGORY = "hs01"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("waist_white_with_hair", "ref_mask", "report")
    DESCRIPTION = ("只把 hs01 出图里的头发贴回腰部白模（不贴生成图的脸 / 身体 / 灰底），作为 Qwen 的图2。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "generated_lora": ("IMAGE", {"tooltip": "hs01 还原到头肩裁切尺寸的出图"}),
            "lora_white": ("IMAGE", {"tooltip": "HS01HeadWaistCrop 的 lora_white（光头白模）"}),
            "lora_mask": ("MASK", {"tooltip": "HS01HeadWaistCrop 的 lora_mask"}),
            "waist_white": ("IMAGE", {"tooltip": "HS01HeadWaistCrop 的 waist_white（工作尺寸）"}),
            "info": ("STRING", {"forceInput": True, "tooltip": "HS01HeadWaistCrop 的 info"}),
            "feather": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1, "tooltip": "贴回边缘羽化像素"}),
            "grow": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.8, "step": 0.02,
                               "tooltip": "头发允许超出光头剪影多少（头宽的比例）"}),
        }}

    def run(self, generated_lora, lora_white, lora_mask, waist_white, info, feather, grow):
        inf = json.loads(info)
        gen = _rgb(generated_lora)
        bald = _rgb(lora_white)
        H, W = bald.shape[:2]
        if gen.shape[:2] != (H, W):
            gen = _resize_rgb(gen, W, H)
        sil = _mask(lora_mask, H, W) > 0.5
        if float(sil.mean()) < 0.001:
            sil = _gray(bald) > 0.03
        head_w = float(inf.get("head_width", W * 0.3))
        grow_px = max(2, int(round(float(grow) * head_w)))
        near = _dilate(sil, grow_px)

        # 只取头发：hs01 的头发是黑的，白模皮肤 / 灰底都亮。用暗部 + 人附近 + 腰部以上，
        # 再扣掉脸椭圆；不要生成图的脸 / 身体（那是另一个人，拼上去会有一圈灰边）。
        dark = _gray(gen) < 0.22
        top_region = np.zeros((H, W), bool)
        ylim = int(np.clip((inf.get("waist_y", H) - inf["lora_box"][1]) * (H / max(1.0, inf["lora_size"][1])), 0, H))
        top_region[:max(1, ylim), :] = True
        fcx = (inf["face_center"][0] - inf["lora_box"][0]) * (W / max(1.0, inf["lora_size"][0]))
        fcy = (inf["face_center"][1] - inf["lora_box"][1]) * (H / max(1.0, inf["lora_size"][1]))
        fw = inf["face_size"][0] * (W / max(1.0, inf["lora_size"][0]))
        fh = inf["face_size"][1] * (H / max(1.0, inf["lora_size"][1]))
        face = _ellipse_mask(H, W, fcx, fcy + 0.16 * fh, 0.42 * fw, 0.30 * fh, 2.0) > 0.3
        hair = dark & near & top_region & (~face)
        hair = _close(hair, 2)
        hair = _largest_components(hair, 0.0008)
        if not hair.any():
            hair = dark & near & top_region
        alpha = _blur(hair.astype(np.float32), max(0.0, float(feather) / 2.0))

        work_w = int(waist_white.shape[2])
        work_h = int(waist_white.shape[1])
        sx, sy = inf["work_scale"]
        lx, ly = inf["lora_box"][:2]
        wx, wy = inf["waist_box"][:2]
        tx = int(round((lx - wx) * sx))
        ty = int(round((ly - wy) * sy))
        tw = max(1, int(round((inf["lora_box"][2] - inf["lora_box"][0]) * sx)))
        th = max(1, int(round((inf["lora_box"][3] - inf["lora_box"][1]) * sy)))
        sx0, sy0 = max(0, tx), max(0, ty)
        sx1, sy1 = min(work_w, tx + tw), min(work_h, ty + th)
        if sx1 <= sx0 or sy1 <= sy0:
            raise ValueError("头肩框不在腰部画布里；检查 crop 输出")

        gen_r = _resize_rgb(gen, tw, th)
        alpha_r = _resize_mask(alpha, tw, th)
        canvas = _rgb(waist_white).copy()
        sub = canvas[sy0:sy1, sx0:sx1].astype(np.float32)
        gsub = gen_r[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx].astype(np.float32)
        asub = alpha_r[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx][..., None]
        canvas[sy0:sy1, sx0:sx1] = np.clip(sub * (1 - asub) + gsub * asub + 0.5, 0, 255).astype(np.uint8)
        ref_mask = np.zeros((work_h, work_w), np.float32)
        ref_mask[sy0:sy1, sx0:sx1] = asub[..., 0]
        report = "\n".join([
            f"头肩出图 {gen.shape[1]}×{gen.shape[0]} → 腰部画布 {work_w}×{work_h} 的 [{sx0},{sy0},{sx1},{sy1}]",
            f"只贴头发：暗部 {int(dark.sum())} px，扣掉脸后 {int(hair.sum())} px；羽化 {feather}px",
        ])
        return _img_t(canvas), _mask_t(ref_mask), report


# ---------------------------------------------------------------------------
# 3. 拼回原图
# ---------------------------------------------------------------------------

class HS01Composite:
    CATEGORY = "hs01"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "crop", "mask", "report")
    DESCRIPTION = ("把 Qwen 出图（1152×1536 / 1152×1440 工作尺寸）缩回腰部原像素，整块按羽化边贴回原图。"
                   "不做头发遮罩 / 人物遮罩，避免额角碎发、旧头发、白底矩形、边缘光晕等遮罩问题；"
                   "Qwen 出图本身已经保留原图的脸和背景。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "qwen_result": ("IMAGE", {"tooltip": "Qwen 出图（1152×1536 / 1152×1440 工作尺寸）"}),
            "waist_photo_native": ("IMAGE", {"tooltip": "HS01HeadWaistCrop 的 waist_photo_native（原像素）"}),
            "full_photo": ("IMAGE", {"tooltip": "整张原照片（跟 ScenePose 同一个 LoadImage）"}),
            "info": ("STRING", {"forceInput": True, "tooltip": "HS01HeadWaistCrop 的 info"}),
            "feather": ("INT", {"default": 64, "min": 0, "max": 256, "step": 2,
                                "tooltip": "裁切框四边羽化像素；白底 / 纯色背景建议 32–64"}),
        }}

    def run(self, qwen_result, waist_photo_native, full_photo, info, feather):
        inf = json.loads(info)
        W, H = inf["full_size"]
        wx, wy, wx1, wy1 = inf["waist_box"]
        w0, h0 = wx1 - wx, wy1 - wy

        orig = _rgb(waist_photo_native)
        if orig.shape[:2] != (h0, w0):
            orig = _resize_rgb(orig, w0, h0)
        q = _rgb(qwen_result)
        q = _resize_rgb(q, w0, h0)

        # 整块裁切 + 四边线性羽化；不做任何内容遮罩。
        f = max(0, min(int(feather), min(w0, h0) // 2))
        mask = np.ones((h0, w0), np.float32)
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f + 1, dtype=np.float32)[1:]
            mask[:f, :] *= ramp[:, None]
            mask[-f:, :] *= ramp[::-1, None]
            mask[:, :f] *= ramp[None, :]
            mask[:, -f:] *= ramp[::-1][None, :]

        out_crop = np.clip(orig.astype(np.float32) * (1 - mask[..., None]) + q.astype(np.float32) * mask[..., None] + 0.5,
                           0, 255).astype(np.uint8)

        full = _rgb(full_photo).copy()
        if full.shape[:2] != (H, W):
            full = _resize_rgb(full, W, H)
        full[wy:wy1, wx:wx1] = out_crop

        report = "\n".join([
            f"腰部原像素 {w0}×{h0}，整块贴回整图 [{wx},{wy},{wx1},{wy1}]",
            f"四边羽化 {f}px；不做头发 / 人物遮罩，避免额角碎发和背景矩形",
            f"Qwen 图缩回 {w0}×{h0}",
        ])
        return _img_t(full), _img_t(out_crop), _mask_t(mask), report


# ---------------------------------------------------------------------------
# 4. 提示词编号映射
# ---------------------------------------------------------------------------

class HS01PromptMap:
    CATEGORY = "hs01"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    DESCRIPTION = ("把用户写的「图1 / 图2」换成 Qwen 2.1 编码器认识的 <image1> / <image2>（PE 关闭时必须）。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"default": "将图2的发型迁移到图1，保持图1人物的面部特征完全不变。Change the hairstyle of the person in <image1> to the high bun updo shown in <image2>: a voluminous bun on top of the head, with long loose strands on both sides.", "multiline": True}),
        }}

    def run(self, text):
        out = text
        for i in (1, 2, 3, 4):
            out = out.replace(f"图{i}", f"<image{i}>").replace(f"图 {i}", f"<image{i}>")
        if "<image1>" not in out:
            out = "<image1> " + out
        report = f"编码器提示词：{out}" if out != text else "提示词里没有「图1/图2」，原样输出"
        return out, report


NODE_CLASS_MAPPINGS = {
    "HS01OneCrop": HS01OneCrop,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HS01OneCrop": "hs02 一次手动裁切（头到腰，内部带头肩）",
}
