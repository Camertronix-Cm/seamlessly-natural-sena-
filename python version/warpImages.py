# -*- coding: utf-8 -*-
"""
warpImages - Gestion du warping et des transformations d'images
"""
import torch
import numpy as np
import cv2
import math


def safe_inv(M):
    try:
        return np.linalg.inv(np.asarray(M, np.float32)).astype(np.float32)
    except Exception:
        return np.eye(3, dtype=np.float32)


def distribution_score(pts, mask, img_shape, grid_size=3):
    inliers = pts[mask]
    if len(inliers) < 3:
        return 0.0
    h, w = img_shape[:2]
    var_x = np.var(inliers[:,0]) / (w**2)
    var_y = np.var(inliers[:,1]) / (h**2)
    dispersion = (var_x + var_y)
    gx, gy = grid_size, grid_size
    cells = set()
    for x, y in inliers:
        cx = int(min(gx-1, gx * x / w))
        cy = int(min(gy-1, gy * y / h))
        cells.add((cx, cy))
    occ_ratio = len(cells) / (gx * gy)
    return 0.5 * dispersion + 0.5 * occ_ratio


def affine_from_pts(src, dst, img_shape, min_pts=3, ransac_thresh=3.0, confidence=0.995, refine_iters=50):
    if src is None or dst is None:
        print("src/dst missing → identity.")
        return np.eye(3, dtype=np.float32), np.zeros(0, dtype=bool)
    src = np.asarray(src, np.float32).reshape(-1, 2)
    dst = np.asarray(dst, np.float32).reshape(-1, 2)
    ok = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src, dst = src[ok], dst[ok]
    n = len(src)
    if n < min_pts:
        print(f"Not enough of points ({n} < {min_pts}) → identity.")
        return np.eye(3, dtype=np.float32), np.zeros(n, dtype=bool)
    Ap, inl_p = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, confidence=confidence, maxIters=refine_iters)
    nin_p = int(inl_p.sum()) if inl_p is not None else 0
    score_p = distribution_score(src, inl_p.ravel().astype(bool), img_shape) if inl_p is not None else 0
    Aa, inl_a = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, confidence=confidence, maxIters=refine_iters)
    nin_a = int(inl_a.sum()) if inl_a is not None else 0
    score_a = distribution_score(src, inl_a.ravel().astype(bool), img_shape) if inl_a is not None else 0
    if nin_a > nin_p:
        A_best, inl_best = Aa, inl_a
    elif nin_p > nin_a:
        A_best, inl_best = Ap, inl_p
    else:
        if score_a > score_p:
            A_best, inl_best = Aa, inl_a
        else:
            print("Equal inliers → Choose Similarity (better or equal distribution)")
            A_best, inl_best = Ap, inl_p
    if A_best is None or inl_best is None:
        return np.eye(3, dtype=np.float32), np.zeros(n, dtype=bool)
    M = np.eye(3, dtype=np.float32)
    M[:2, :] = A_best.astype(np.float32)
    return M, inl_best.ravel().astype(bool)


def poly_clip_to_rect(poly, w, h):
    def inside(P, eid):
        x,y=P
        if eid==0: return x>=0
        if eid==1: return x<=w-1
        if eid==2: return y>=0
        if eid==3: return y<=h-1
    def inter(A,B,eid):
        Ax,Ay=A; Bx,By=B
        if eid==0: t=(0-Ax)/(Bx-Ax+1e-12); return (0, Ay+t*(By-Ay))
        if eid==1: t=((w-1)-Ax)/(Bx-Ax+1e-12); return (w-1, Ay+t*(By-Ay))
        if eid==2: t=(0-Ay)/(By-Ay+1e-12); return (Ax+t*(Bx-Ax), 0)
        if eid==3: t=((h-1)-Ay)/(By-Ay+1e-12); return (Ax+t*(Bx-Ax), h-1)
    P = [tuple(p) for p in poly]
    for eid in range(4):
        out=[]
        for i in range(len(P)):
            A=P[i]; B=P[(i+1)%len(P)]
            if inside(B,eid):
                if inside(A,eid): out.append(B)
                else: out.append(inter(A,B,eid)); out.append(B)
            elif inside(A,eid):
                out.append(inter(A,B,eid))
        P=out
        if not P: break
    return np.array(P, dtype=np.float32)


def mask_from_polygon_size(size_hw, poly):
    h,w = size_hw
    mask = np.zeros((h,w), dtype=np.uint8)
    if poly.size>0: cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def smootherstep01(t):
    return t*t*(t*(t*6 - 15) + 10)


def make_target_ramp(mask_tgt, band_ratio=0.10):
    H, W = mask_tgt.shape[:2]
    inside = (mask_tgt > 0).astype(np.uint8)
    if inside.max() == 0: return np.zeros((H,W), np.float32)
    diag = float(np.hypot(W, H))
    band_px = max(40, int(band_ratio * diag))
    dist_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5).astype(np.float32)
    dist_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5).astype(np.float32)
    sdf = dist_in - dist_out
    t = np.clip(sdf / float(max(1, band_px)), 0.0, 1.0)
    return smootherstep01(t).astype(np.float32)


def affine_diagnostics(A, src_pts, dst_pts, Aglob=None, cell_bbox=None):
    A = np.asarray(A, np.float32)
    src = np.asarray(src_pts, np.float32).reshape(-1, 2)
    dst = np.asarray(dst_pts, np.float32).reshape(-1, 2)
    n = len(src)
    if n >= 1:
        S_h = np.hstack([src, np.ones((n,1), np.float32)])
        Ph = (S_h @ A.T)
        z = np.clip(Ph[:, 2:3], 1e-9, None)
        pred = Ph[:, :2] / z
        rmse = float(np.sqrt(np.mean(np.sum((pred - dst)**2, axis=1))))
    else:
        rmse = 1e9
    A2 = A[:2, :2]
    try:
        _, s, _ = np.linalg.svd(A2.astype(np.float32))
        cond = float(s[0] / max(s[-1], 1e-9))
    except Exception:
        cond = 1e9
    det = float(np.linalg.det(A2.astype(np.float32)))
    delta_mean = 0.0
    if Aglob is not None and cell_bbox is not None:
        x0, x1, y0, y1 = cell_bbox
        xs = np.linspace(x0, x1, num=5, dtype=np.float32)
        ys = np.linspace(y0, y1, num=5, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        P = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3)
        Minv = safe_inv(A)
        Ginv = safe_inv(np.asarray(Aglob, np.float32))
        Pl = (P @ Minv.T); zl = np.clip(Pl[:,2:3], 1e-9, None); Pl_xy = Pl[:, :2] / zl
        Pg = (P @ Ginv.T); zg = np.clip(Pg[:,2:3], 1e-9, None); Pg_xy = Pg[:, :2] / zg
        delta_mean = float(np.mean(np.linalg.norm(Pl_xy - Pg_xy, axis=1)))
    return rmse, n, det, cond, delta_mean


def fit_affine_ridge(src_pts, dst_pts, Aglob, lam=1.8, weights=None):
    src = np.asarray(src_pts, np.float32).reshape(-1,2)
    dst = np.asarray(dst_pts, np.float32).reshape(-1,2)
    n = len(src)
    if n < 3:
        M = np.eye(3, dtype=np.float32); M[:2,:] = Aglob[:2,:]; return M
    X = np.hstack([src, np.ones((n,1),np.float32)])
    if weights is None:
        W = np.eye(n, dtype=np.float32)
    else:
        w = np.clip(weights.astype(np.float32).ravel(), 1e-6, None)
        W = np.diag(w)
    XtWX = X.T @ W @ X
    I = np.eye(3, dtype=np.float32)
    Ag = Aglob[:2,:]
    A_est = np.zeros((2,3), np.float32)
    for dim in range(2):
        y = dst[:,dim:dim+1]
        ag = Ag[dim:dim+1,:].T
        lhs = XtWX + lam * I
        rhs = X.T @ W @ y + lam * ag
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        A_est[dim,:] = sol.ravel()
    M = np.eye(3, dtype=np.float32); M[:2,:] = A_est
    return M


def refit_cell_affine(cell, src_pts, dst_pts, Wt, Ht, Aglob, neigh_cells=None, lam=2.2):
    cx, cy = cell["center"]
    dst_int = np.round(dst_pts).astype(int)
    keep = (dst_int[:,0]>=0) & (dst_int[:,0]<Wt) & (dst_int[:,1]>=0) & (dst_int[:,1]<Ht)
    def collect(mask):
        idx = np.where(keep & (mask[dst_int[:,1], dst_int[:,0]] > 0))[0]
        return src_pts[idx], dst_pts[idx]
    all_sp, all_dp = [], []
    sp, dp = collect(cell["mask"]); all_sp.append(sp); all_dp.append(dp)
    if neigh_cells:
        for nc in neigh_cells:
            spn, dpn = collect(nc["mask"])
            if len(spn) > 0: all_sp.append(spn); all_dp.append(dpn)
    valid_sp = [a for a in all_sp if len(a) > 0]
    valid_dp = [a for a in all_dp if len(a) > 0]
    if len(valid_sp) == 0 or len(valid_dp) == 0:
        M = np.eye(3, dtype=np.float32); M[:2,:] = Aglob[:2,:]; return M, 0.2
    SP = np.vstack(valid_sp).astype(np.float32)
    DP = np.vstack(valid_dp).astype(np.float32)
    if len(SP) < 3:
        M = np.eye(3, dtype=np.float32); M[:2,:] = Aglob[:2,:]; return M, 0.2
    d2 = (DP[:,0]-cx)**2 + (DP[:,1]-cy)**2
    x0,x1,y0,y1 = cell["bbox"]
    sigma = 0.35 * np.hypot(max(1,x1-x0), max(1,y1-y0))
    w = np.exp(-d2/(2.0*sigma*sigma)).astype(np.float32)
    M = fit_affine_ridge(SP, DP, Aglob, lam=float(lam), weights=w)
    conf = float((w.sum() / (w.max()+1e-9)) / 20.0)
    conf = max(0.2, min(1.5, conf))
    return M, conf


def build_canvas_geometry(src_shape, tgt_shape, Aglob, Hhint=None, margin_ratio=0.12, min_pad=96):
    Hs, Ws = src_shape[:2]; Ht, Wt = tgt_shape[:2]
    diag_s = math.hypot(Ws, Hs)
    pad = int(max(min_pad, margin_ratio * diag_s))
    src_poly = np.float32([[0,0],[Ws-1,0],[Ws-1,Hs-1],[0,Hs-1]])
    sp_h = np.hstack([src_poly, np.ones((4,1),dtype=np.float32)])
    xs = [0, Wt-1, Wt-1, 0]; ys = [0, 0, Ht-1, Ht-1]
    tp_g = (sp_h @ Aglob.T); tp_g = tp_g[:,:2]/np.clip(tp_g[:,2:3], 1e-9, None)
    xs += tp_g[:,0].tolist(); ys += tp_g[:,1].tolist()
    if Hhint is not None:
        tp_h = (sp_h @ Hhint.T); tp_h = tp_h[:,:2]/np.clip(tp_h[:,2:3], 1e-9, None)
        xs += tp_h[:,0].tolist(); ys += tp_h[:,1].tolist()
    minx, maxx = math.floor(np.min(xs)) - pad, math.ceil(np.max(xs)) + pad
    miny, maxy = math.floor(np.min(ys)) - pad, math.ceil(np.max(ys)) + pad
    ox = -min(0, minx); oy = -min(0, miny)
    Wc = int(maxx - minx + 1); Hc = int(maxy - miny + 1)
    return Wc, Hc, int(ox), int(oy)


def grid_cells_from_mask(mask, grid=(1,1)):
    H, W = mask.shape[:2]
    ys, xs = np.where(mask >0)
    if len(xs)==0:
        xmin, xmax, ymin, ymax = 0, W-1, 0, H-1
    else:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
    padding = 30
    xmin = max(0, xmin - padding); xmax = min(W-1, xmax + padding)
    ymin = max(0, ymin - padding); ymax = min(H-1, ymax + padding)
    gx, gy = grid
    cells = []
    for j in range(gy):
        for i in range(gx):
            x0 = int(round(xmin + (xmax -xmin) * i/gx))
            x1 = int(round(xmin + (xmax-xmin) * (i+1)/gx))
            y0 = int(round(ymin + (ymax-ymin) * j/gy))
            y1 = int(round(ymin + (ymax-ymin) * (j+1)/gy))
            cmask = np.zeros((H,W), np.uint8); cmask[y0:y1, x0:x1] = 255
            cmask = cv2.bitwise_and(cmask, mask) if mask.max() >0 else cmask
            m = cv2.moments(cmask, binaryImage=True)
            if m["m00"] >0: cx = m["m10"]/m["m00"]; cy = m["m01"]/m["m00"]
            else: cx = 0.5*(x0+x1); cy = 0.5*(y0+y1)
            cells.append({"mask": cmask, "bbox": (x0,x1,y0,y1), "center": (float(cx), float(cy))})
    return cells


def build_field_ffd(baseH, models, cells, cell_conf, Wc, Hc, ox, oy, Ht, Wt, overlap_mask_tgt, lattice_hw=(64,64), sigma_ratio=0.30, scale=0.50, dmax=60.0):
    Ny, Nx = lattice_hw
    yy_l, xx_l = np.mgrid[0:Ny, 0:Nx].astype(np.float32)
    xx_t_l = (xx_l / max(1,Nx-1)) * (Wc-1)
    yy_t_l = (yy_l / max(1,Ny-1)) * (Hc-1)
    Yh_l = np.stack([xx_t_l - ox, yy_t_l - oy, np.ones_like(xx_t_l)], axis=-1)
    BaseInv = safe_inv(baseH)
    Bv = (Yh_l.reshape(-1,3) @ BaseInv.T).reshape(Ny,Nx,3)
    zb = np.clip(Bv[...,2:3], 1e-9, None)
    bx = (Bv[...,0]/zb[...,0]).astype(np.float32)
    by = (Bv[...,1]/zb[...,0]).astype(np.float32)
    centers = np.float32([c["center"] for c in cells])
    if len(centers) == 0:
        return np.zeros((Hc,Wc), np.float32), np.zeros((Hc,Wc), np.float32), np.zeros((Hc,Wc), np.float32)
    cx = centers[:,0][:,None,None]
    cy = centers[:,1][:,None,None]
    dist2 = (xx_t_l[None,:,:] - cx)**2 + (yy_t_l[None,:,:] - cy)**2
    cell_diags = np.array([math.hypot(c["bbox"][1]-c["bbox"][0], c["bbox"][3]-c["bbox"][2]) for c in cells], np.float32)
    sigma = sigma_ratio * (cell_diags.mean() + 1e-6)
    w_raw = np.exp(-dist2 / (2.0 * (sigma**2)))
    conf = np.clip(np.asarray(cell_conf, np.float32), 0.2, None)[:,None,None]
    w_raw *= conf
    Wsum = np.clip(w_raw.sum(axis=0, keepdims=True), 1e-6, None)
    w_norm = w_raw / Wsum
    dx_l = np.zeros((Ny,Nx), np.float32)
    dy_l = np.zeros((Ny,Nx), np.float32)
    Yh_flat = Yh_l.reshape(-1,3)
    for k, Mk in enumerate(models):
        Minv = safe_inv(Mk)
        Lv = (Yh_flat @ Minv.T).reshape(Ny,Nx,3)
        zl = np.clip(Lv[...,2:3], 1e-9, None)
        xk = (Lv[...,0]/zl[...,0]).astype(np.float32)
        yk = (Lv[...,1]/zl[...,0]).astype(np.float32)
        dx_l += w_norm[k] * (xk - bx)
        dy_l += w_norm[k] * (yk - by)
    np.clip(dx_l, -dmax, dmax, out=dx_l)
    np.clip(dy_l, -dmax, dmax, out=dy_l)
    dx_l = cv2.GaussianBlur(dx_l, (0,0), 0.6)
    dy_l = cv2.GaussianBlur(dy_l, (0,0), 0.6)
    ramp_tgt = make_target_ramp(overlap_mask_tgt, band_ratio=0.10)
    ramp_canvas = np.zeros((Hc, Wc), np.float32)
    y0, y1 = oy, oy + Ht; x0, x1 = ox, ox + Wt
    if y0 < Hc and x0 < Wc and y1 > 0 and x1 > 0:
        yy0 = max(0, y0); yy1 = min(Hc, y1)
        xx0 = max(0, x0); xx1 = min(Wc, x1)
        ry0 = yy0 - oy; ry1 = yy1 - oy
        rx0 = xx0 - ox; rx1 = xx1 - ox
        ramp_canvas[yy0:yy1, xx0:xx1] = ramp_tgt[ry0:ry1, rx0:rx1]
    dx_full = cv2.resize(dx_l, (Wc, Hc), interpolation=cv2.INTER_CUBIC)
    dy_full = cv2.resize(dy_l, (Wc, Hc), interpolation=cv2.INTER_CUBIC)
    dx_full = (scale * dx_full * ramp_canvas).astype(np.float32)
    return dx_full, dy_full, ramp_canvas


def make_match_density_canvas(dst_pts, Wc, Hc, ox, oy, sigma_px=24):
    den = np.zeros((Hc, Wc), np.float32)
    if dst_pts is None or len(dst_pts)==0:
        return den
    pts = np.asarray(dst_pts, np.float32)
    xs = np.clip(np.round(ox + pts[:,0]).astype(int), 0, Wc-1)
    ys = np.clip(np.round(oy + pts[:,1]).astype(int), 0, Hc-1)
    den[ys, xs] += 1.0
    k = int(max(15, 6*int(max(1, sigma_px)))); k |= 1
    den = cv2.GaussianBlur(den, (k,k), sigma_px)
    m = den.max()
    if m > 1e-9: den /= m
    return den


def apply_seam_guard(dx_full, dy_full, ramp_canvas, dst_pts, Wc, Hc, ox, oy, min_gate=0.30, blur_sigma=1.1):
    taper = smootherstep01(np.clip(ramp_canvas, 0.0, 1.0)) ** 1.2
    density = make_match_density_canvas(dst_pts, Wc, Hc, ox, oy, sigma_px=24)
    gate = min_gate + (1.0 - min_gate) * smootherstep01(np.clip(density, 0.0, 1.0))
    mask = (taper * gate).astype(np.float32)
    dx = cv2.GaussianBlur((dx_full * mask).astype(np.float32), (0,0), blur_sigma)
    dy = cv2.GaussianBlur((dy_full * mask).astype(np.float32), (0,0), blur_sigma)
    return dx, dy, mask, density


def warp_images_with_xfeat_points(src_img, tgt_img, mkpts_0, mkpts_1):
    if src_img is None or tgt_img is None:
        raise ValueError("Images invalides")
    if len(mkpts_0) < 12 or len(mkpts_1) < 12:
        raise ValueError("Pas assez de points matches (minimum 12)")
    Hs, Ws = src_img.shape[:2]
    Ht, Wt = tgt_img.shape[:2]
    src_pts = np.asarray(mkpts_0, dtype=np.float32)
    dst_pts = np.asarray(mkpts_1, dtype=np.float32)
    Aglob, inliers_mask = affine_from_pts(src_pts, dst_pts, src_img.shape)
    src_pts = src_pts[inliers_mask]
    dst_pts = dst_pts[inliers_mask]
    BaseInv = safe_inv(Aglob)
    src_poly = np.float32([[0,0],[Ws-1,0],[Ws-1,Hs-1],[0,Hs-1]])
    src_poly_w = cv2.perspectiveTransform(src_poly.reshape(-1,1,2), Aglob).reshape(-1,2)
    poly_tgt = poly_clip_to_rect(src_poly_w.tolist(), Wt, Ht)
    overlap_mask_tgt = mask_from_polygon_size((Ht,Wt), poly_tgt)
    if len(poly_tgt) > 0:
        poly_src = cv2.perspectiveTransform(poly_tgt.reshape(-1,1,2), BaseInv).reshape(-1,2)
        overlap_mask_src = mask_from_polygon_size(src_img.shape[:2], poly_src)
    else:
        overlap_mask_src = np.zeros(src_img.shape[:2], dtype=np.uint8)
    cells = grid_cells_from_mask(overlap_mask_tgt, grid=(2,2))
    models, cell_conf = [], []
    dst_int_all = np.round(dst_pts).astype(int)
    for ci, c in enumerate(cells):
        cmask = c["mask"]
        keep = (dst_int_all[:,0] >=0) & (dst_int_all[:,0] <Wt) & (dst_int_all[:,1] >=0) & (dst_int_all[:,1] <Ht)
        keep &= (cmask[dst_int_all[:,1], dst_int_all[:,0]] > 0)
        sp = src_pts[keep]; dp = dst_pts[keep]
        neigh = [c2 for cj, c2 in enumerate(cells) if cj != ci]
        Mk, conf = refit_cell_affine(c, src_pts, dst_pts, Wt, Ht, Aglob, neigh_cells=neigh, lam=2.2)
        rmse, n, det, cond, dmean = affine_diagnostics(Mk, sp, dp, Aglob=Aglob, cell_bbox=c["bbox"])
        bad = (n < 6) or (rmse > 5.0) or (abs(det) < 0.05) or (cond > 120.0) or (dmean > 0.22*np.hypot(Wt,Ht)/4.0)
        if bad:
            Mk2, conf2 = refit_cell_affine(c, src_pts, dst_pts, Wt, Ht, Aglob, neigh_cells=neigh, lam=2.8)
            rmse2, n2, det2, cond2, dmean2 = affine_diagnostics(Mk2, sp, dp, Aglob=Aglob, cell_bbox=c["bbox"])
            score1 = rmse + 0.01*cond + 0.5*max(0.0,0.1-abs(det)) + 0.002*dmean
            score2 = rmse2 + 0.01*cond2 + 0.5*max(0.0,0.1-abs(det2)) + 0.002*dmean2
            if score2 < score1: Mk, conf = Mk2, conf2
        models.append(Mk.astype(np.float32))
        cell_conf.append(conf)
    Wc, Hc, ox, oy = build_canvas_geometry(src_img.shape, tgt_img.shape, Aglob, Hhint=None, margin_ratio=0.12, min_pad=96)
    dx_full, dy_full, ramp_canvas = build_field_ffd(Aglob, models, cells, cell_conf, Wc, Hc, ox, oy, Ht, Wt, overlap_mask_tgt, lattice_hw=(64,64), sigma_ratio=0.35, scale=0.50, dmax=60.0)
    dx_full, dy_full, mask_seam, density = apply_seam_guard(dx_full, dy_full, ramp_canvas, dst_pts, Wc, Hc, ox, oy, min_gate=0.30, blur_sigma=1.1)
    yy_c, xx_c = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    xx_t = xx_c - ox; yy_t = yy_c - oy
    Yh = np.stack([xx_t, yy_t, np.ones_like(xx_t)], axis=-1)
    BaseInvY = (Yh.reshape(-1,3) @ BaseInv.T).reshape(Hc,Wc,3)
    z0 = np.clip(BaseInvY[...,2:3], 1e-9, None)
    map0_x = (BaseInvY[...,0]/z0[...,0]).astype(np.float32)
    map0_y = (BaseInvY[...,1]/z0[...,0]).astype(np.float32)
    map_x = (map0_x + dx_full).astype(np.float32)
    map_y = (map0_y + dy_full).astype(np.float32)
    listA = []
    listB = []
    for pt in dst_pts:
        x, y = pt
        xc = x + ox
        yc = y + oy
        if 0 <= xc < Wc and 0 <= yc < Hc:
            listB.append([float(xc), float(yc)])
        else:
            listB.append([float('nan'), float('nan')])
    src_pts_h = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=np.float32)])
    src_pts_t = (src_pts_h @ Aglob.T)
    z = np.clip(src_pts_t[:, 2], 1e-9, None)
    src_pts_proj = src_pts_t[:, :2] / z[:, None]
    src_pts_c = src_pts_proj + np.array([[ox, oy]], dtype=np.float32)
    for i, (xc, yc) in enumerate(src_pts_c):
        if not (0 <= xc < Wc and 0 <= yc < Hc):
            listA.append([float('nan'), float('nan')])
            continue
        x0, y0 = int(xc), int(yc)
        x1, y1 = min(x0 + 1, Wc - 1), min(y0 + 1, Hc - 1)
        dx00, dx01 = dx_full[y0, x0], dx_full[y0, x1]
        dx10, dx11 = dx_full[y1, x0], dx_full[y1, x1]
        dy00, dy01 = dy_full[y0, x0], dy_full[y0, x1]
        dy10, dy11 = dy_full[y1, x0], dy_full[y1, x1]
        wx, wy = xc - x0, yc - y0
        dx = (dx00 * (1 - wx) + dx01 * wx) * (1 - wy) + (dx10 * (1 - wx) + dx11 * wx) * wy
        dy = (dy00 * (1 - wx) + dy01 * wx) * (1 - wy) + (dy10 * (1 - wx) + dy11 * wx) * wy
        final_x = xc + dx
        final_y = yc + dy
        listA.append([float(final_x), float(final_y)])
    canvasA = cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    canvasB = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    y0, y1 = oy, oy + Ht; x0, x1 = ox, ox + Wt
    if y0 < Hc and x0 < Wc and y1 > 0 and x1 > 0:
        yy0 = max(0, y0); yy1 = min(Hc, y1)
        xx0 = max(0, x0); xx1 = min(Wc, x1)
        src_y0 = yy0 - oy; src_y1 = yy1 - oy
        src_x0 = xx0 - ox; src_x1 = xx1 - ox
        canvasB[yy0:yy1, xx0:xx1] = tgt_img[src_y0:src_y1, src_x0:src_x1]
    return canvasA, canvasB, listA, listB, Aglob, ox, oy, (Hs, Ws), (Ht, Wt)


def safe_extract_transformed_points(data_list, index_list):
    if not data_list:
        return []
    try:
        target_list = data_list[index_list]
    except (IndexError, TypeError):
        return []
    if not target_list:
        return []
    transformed = []
    ignored_count = 0
    seq_types = (list, tuple, np.ndarray)
    def _has_two_coords(pt):
        if not isinstance(pt, seq_types):
            return False
        try:
            size = (pt.size if isinstance(pt, np.ndarray) else len(pt))
            return size >= 2
        except Exception:
            return False
    def _xy_as_float(pt):
        arr = np.asarray(pt).reshape(-1)
        x, y = float(arr[0]), float(arr[1])
        return x, y
    for item in target_list:
        if not isinstance(item, seq_types) or len(item) < 2:
            ignored_count += 1
            continue
        pt1, pt2 = item[0], item[1]
        if not _has_two_coords(pt1):
            ignored_count += 1
            continue
        if not _has_two_coords(pt2):
            ignored_count += 1
            continue
        try:
            x1, y1 = _xy_as_float(pt1)
            x2, y2 = _xy_as_float(pt2)
            transformed.append([(x1, y1), (x2, y2)])
        except Exception:
            ignored_count += 1
            continue
    return transformed


def pair_points_list(flat_list):
    if len(flat_list) % 2 != 0:
        flat_list = flat_list[:-1]
    return [[flat_list[i], flat_list[i+1]] for i in range(0, len(flat_list), 2)]