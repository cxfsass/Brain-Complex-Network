# -*- coding: utf-8 -*-
"""
EEG .npy 交互式浏览器（图嵌入 Tkinter GUI，不弹出外部窗口）
新增功能（面向清醒/疲劳分析）：
1) Alert vs Fatigue 并排对比模式（同一被试同一参数）
2) RMS 显示模式：Alert / Fatigue / ΔRMS(Fatigue-Alert)
3) 自动选取“代表性 epoch”（基于 epoch RMS 中位数）
4) 单通道界面加入频段能量速览（δ/θ/α/β）
"""

import os
import re
import numpy as np
try:
    from scipy.signal import welch
except Exception:
    welch = None  # 若无 scipy，将退回 FFT 能量估计（不推荐但可用）
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
# ======== 关键：设置中文字体（按可用性依次尝试）========
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",  # 微软雅黑（Windows 常见，推荐）
    "SimHei",           # 黑体（Windows 常见）
    "SimSun",           # 宋体（Windows 常见）
    "Arial Unicode MS"  # mac/部分环境
]
matplotlib.rcParams["axes.unicode_minus"] = False  # 避免负号显示异常
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

ALERT_COLOR = "#E91E63"
FATIGUE_COLOR = "#1F77B4"
DELTA_POS_COLOR = "#D32F2F"
DELTA_NEG_COLOR = "#1976D2"


def scan_subject_files(root_dir: str):
    """
    扫描目录并解析被试与 session（EEG1/EEG4）
    文件名示例：12345678910_byKKf_EEG1.npy
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"目录不存在：{root_dir}")

    pat = re.compile(r"^(?P<sub>.+?)_.*_(?P<sess>EEG1|EEG4)\.npy$", re.IGNORECASE)
    subjects = {}

    for fn in os.listdir(root_dir):
        if not fn.lower().endswith(".npy"):
            continue
        m = pat.match(fn)
        if not m:
            continue
        sub = m.group("sub")
        sess = m.group("sess").upper()
        subjects.setdefault(sub, {})
        subjects[sub][sess] = os.path.join(root_dir, fn)

    return {k: v for k, v in subjects.items() if len(v) > 0}


def _safe_load_npy(path: str):
    """安全读取 .npy，失败抛出异常。"""
    return np.load(path, allow_pickle=False)


def _epoch_rms_vector(eeg: np.ndarray) -> np.ndarray:
    """
    计算每个 epoch 的 RMS（对通道与时间点做均方根）
    eeg: shape (E, C, T)
    return: shape (E,)
    """
    return np.sqrt(np.nanmean(eeg ** 2, axis=(1, 2)))


def _representative_epoch_idx(eeg: np.ndarray) -> int:
    """
    代表性 epoch：RMS 最接近中位数的 epoch
    """
    rms_vec = _epoch_rms_vector(eeg)
    med = np.nanmedian(rms_vec)
    idx = int(np.nanargmin(np.abs(rms_vec - med)))
    return idx


def _representative_epoch_idx_with_mask(eeg: np.ndarray, mask: np.ndarray) -> int:
    """
    带掩码的代表性 epoch：仅在 mask=True 的 epoch 中选择
    """
    if mask is None:
        return _representative_epoch_idx(eeg)
    rms_vec = _epoch_rms_vector(eeg)
    if mask.shape[0] != rms_vec.shape[0]:
        return _representative_epoch_idx(eeg)
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return _representative_epoch_idx(eeg)
    med = np.nanmedian(rms_vec[valid])
    idx_local = int(np.nanargmin(np.abs(rms_vec[valid] - med)))
    return int(valid[idx_local])


def _bandpowers_fft(x: np.ndarray, fs: int):
    """
    计算单通道信号的频段能量（基于 rFFT 的简洁实现）
    - 注释：用于 GUI 速览；若后续要严谨谱估计可改 Welch
    返回：dict {band: power}
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("bandpowers 输入必须是一维信号")

    # 去直流，减少低频偏置
    x = x - np.nanmean(x)

    n = len(x)
    if n < 8:
        return {"delta": np.nan, "theta": np.nan, "alpha": np.nan, "beta": np.nan}

    # rFFT 功率谱（单位：幅度^2）
    X = np.fft.rfft(x)
    P = (np.abs(X) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    def band_int(f_lo, f_hi):
        m = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(m):
            return 0.0
        # 频率积分近似：功率谱求和 * df
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        return float(np.nansum(P[m]) * df)

    return {
        "delta": band_int(0.5, 4.0),
        "theta": band_int(4.0, 8.0),
        "alpha": band_int(8.0, 13.0),
        "beta":  band_int(13.0, 30.0),
    }



def _bandpower_welch_matrix(eeg: np.ndarray, fs: int, bands=None, nperseg: int = 1024, noverlap: int = 512):
    """
    计算多 epoch、多通道 EEG 的频段能量矩阵（Welch）
    输入：
    - eeg: shape (E, C, T)
    输出：
    - band_powers: dict {band: (E,C)}，单位为功率积分（线性域）
    说明：
    - Welch 更适合论文级谱估计；若 scipy 不可用则退回到 FFT 简化实现
    """
    if bands is None:
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta":  (13.0, 30.0),
        }

    eeg = np.asarray(eeg, dtype=float)
    if eeg.ndim != 3:
        raise ValueError(f"eeg 必须是 3D (E,C,T)，当前 shape={eeg.shape}")

    E, C, T = eeg.shape
    if welch is None:
        # 退回 FFT：逐通道逐 epoch（速度慢但保证可用）
        out = {k: np.full((E, C), np.nan, dtype=float) for k in bands.keys()}
        for e in range(E):
            for c in range(C):
                bp = _bandpowers_fft(eeg[e, c, :], fs)
                for k in out.keys():
                    out[k][e, c] = bp.get(k, np.nan)
        return out

    # Welch：把 (E,C,T) reshape 成 (E*C, T) 一次性跑
    X = eeg.reshape(E * C, T)
    # 去直流
    X = X - np.nanmean(X, axis=1, keepdims=True)

    f, Pxx = welch(
        X,
        fs=fs,
        nperseg=min(nperseg, T),
        noverlap=min(noverlap, max(0, min(nperseg, T) // 2)),
        axis=-1,
        detrend="constant",
        scaling="density",
    )  # Pxx: (E*C, F)

    df = f[1] - f[0] if len(f) > 1 else 1.0

    out = {}
    for band, (f_lo, f_hi) in bands.items():
        m = (f >= f_lo) & (f < f_hi)
        if not np.any(m):
            out[band] = np.full((E, C), np.nan, dtype=float)
            continue
        bp = np.nansum(Pxx[:, m], axis=1) * df  # 频带积分
        out[band] = bp.reshape(E, C)

    return out


def _robust_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    鲁棒标准化：median/MAD
    - 注释：论文展示“形态”更稳，抗尖峰伪迹
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad)


def plot_stacked_paper_on_ax(
    ax,
    eeg_epoch: np.ndarray,      # (C, T)
    fs: int,
    channel_names=None,
    title: str = None,
    mode: str = "robust_z",     # "raw_offset" / "zscore" / "robust_z"
    offset: float = None,
    linewidth: float = 0.7,
):
    """
    在指定 ax 上绘制“论文级”EEG 多通道叠加图（stacked plot）
    - 注释：适合嵌入 GUI Canvas；不创建新 figure
    """
    if eeg_epoch.ndim != 2:
        raise ValueError(f"eeg_epoch 必须是 2D (C,T)，当前 shape={eeg_epoch.shape}")

    C, T = eeg_epoch.shape
    if channel_names is None:
        channel_names = [f"Ch{c}" for c in range(C)]
    else:
        if len(channel_names) != C:
            raise ValueError(f"channel_names 长度={len(channel_names)} 与 C={C} 不一致")

    t = np.arange(T) / fs
    X = eeg_epoch.astype(float).copy()

    if mode == "zscore":
        for c in range(C):
            mu = np.mean(X[c])
            sd = np.std(X[c]) + 1e-8
            X[c] = (X[c] - mu) / sd
        step = 3.0
    elif mode == "robust_z":
        for c in range(C):
            X[c] = _robust_zscore(X[c])
        step = 3.0
    elif mode == "raw_offset":
        if offset is None:
            per_ch_std = np.std(X, axis=1)
            base = np.median(per_ch_std)
            offset = 4.0 * base if base > 0 else 1.0
        step = float(offset)
    else:
        raise ValueError("mode 只能是 raw_offset / zscore / robust_z")

    ax.clear()
    for c in range(C):
        y = X[c] + (C - 1 - c) * step
        ax.plot(t, y, linewidth=linewidth)

    yticks = [(C - 1 - c) * step for c in range(C)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(channel_names, fontsize=9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")

    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.35)


class EEGGuiBrowserEmbed:
    def __init__(self, root_dir: str, fs: int = 1000):
        self.root_dir = root_dir
        self.fs = fs
        self.subjects = scan_subject_files(root_dir)
        self._has_loaded = False
        self._last_selected_index = 0

        if len(self.subjects) == 0:
            raise RuntimeError("未识别到 *_EEG1.npy / *_EEG4.npy 文件，请检查命名或目录。")

        # 通道名：按你提供的 0-30 顺序
        self.channel_names = [
            "A2","T6","TP8","T4","FT8","F8","O2","P4","CP4","C4","FC4","F4","FP2",
            "OZ","PZ","CPZ","CZ","FCZ","FZ","FP1","F3","FC3","C3","CP3","P3","O1",
            "F7","FT7","T3","TP7","T5"
        ]

        # 缓存：避免频繁切换时重复读盘
        self._cache = {}  # key: (sid, sess) -> eeg ndarray

        self.root = tk.Tk()
        self.root.title("EEG NPY Browser (Embedded Plots)")
        try:
            self.root.option_add("*Font", ("Microsoft YaHei", 10))
        except Exception:
            pass

        # ========== 顶部布局：左（被试） + 右（控制） ==========
        top = ttk.Frame(self.root, padding=10)
        top.grid(row=0, column=0, sticky="nsew")

        # 左：被试列表
        left = ttk.Frame(top)
        left.grid(row=0, column=0, sticky="nsw")

        ttk.Label(left, text="被试列表（点击选择）").grid(row=0, column=0, sticky="w")
        self.listbox = tk.Listbox(left, height=16, width=34)
        self.listbox.grid(row=1, column=0, sticky="nsw", pady=(6, 0))

        self.sub_ids = sorted(self.subjects.keys())
        for sid in self.sub_ids:
            has1 = "EEG1" in self.subjects[sid]
            has4 = "EEG4" in self.subjects[sid]
            tag1 = "1" if has1 else "-"
            tag4 = "4" if has4 else "-"
            self.listbox.insert(tk.END, f"{sid}   [EEG{tag1}/EEG{tag4}]")

        self.listbox.selection_set(0)
        self._last_selected_index = 0

        # 右：控制区
        right = ttk.Frame(top)
        right.grid(row=0, column=1, sticky="nsew", padx=(16, 0))

        ttk.Label(right, text=f"数据目录：{self.root_dir}").grid(row=0, column=0, sticky="w")

        mode_frame = ttk.LabelFrame(right, text="模式选择", padding=(10, 6))
        mode_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.compare_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_frame, text="Alert vs Fatigue 对比模式", variable=self.compare_var).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        ttk.Label(mode_frame, text="Session（非对比）：").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.session_var = tk.StringVar(value="EEG1")
        ttk.Radiobutton(mode_frame, text="EEG1 (alert)", value="EEG1", variable=self.session_var).grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="EEG4 (fatigue)", value="EEG4", variable=self.session_var).grid(row=3, column=0, sticky="w")

        preview_frame = ttk.LabelFrame(right, text="预览参数", padding=(10, 6))
        preview_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(preview_frame, text="Epoch idx").grid(row=0, column=0, sticky="w")
        self.epoch_entry = ttk.Entry(preview_frame, width=10)
        self.epoch_entry.insert(0, "0")
        self.epoch_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(preview_frame, text="Channel idx").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.channel_entry = ttk.Entry(preview_frame, width=10)
        self.channel_entry.insert(0, "0")
        self.channel_entry.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.data_info_label = ttk.Label(preview_frame, text="数据范围：未加载")
        self.data_info_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.rep_btn = ttk.Button(preview_frame, text="使用代表性 epoch", command=self.on_pick_representative_epoch)
        self.rep_btn.grid(row=3, column=0, sticky="w", pady=(8, 0))

        self.load_btn = ttk.Button(preview_frame, text="加载并预览", command=self.on_preview)
        self.load_btn.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        action_frame = ttk.Frame(preview_frame)
        action_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.path_btn = ttk.Button(action_frame, text="查看文件路径", command=self.on_show_path)
        self.path_btn.grid(row=0, column=0, sticky="w")
        self.reset_btn = ttk.Button(action_frame, text="重置设置", command=self.on_reset_controls)
        self.reset_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))

        stats_frame = ttk.LabelFrame(right, text="统计/过滤", padding=(10, 6))
        stats_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(stats_frame, text="RMS 显示：").grid(row=0, column=0, sticky="w")
        self.rms_mode = tk.StringVar(value="Alert")
        self.rms_combo = ttk.Combobox(
            stats_frame,
            textvariable=self.rms_mode,
            values=["Alert", "Fatigue", "ΔRMS(F-A)"],
            width=12,
            state="readonly",
        )
        self.rms_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.use_robust_epochs = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            stats_frame, text="代表性多 epoch", variable=self.use_robust_epochs
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(stats_frame, text="Epoch 数").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.epoch_pool_entry = ttk.Entry(stats_frame, width=10)
        self.epoch_pool_entry.insert(0, "20")
        self.epoch_pool_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.enable_artifact_filter = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            stats_frame, text="伪迹过滤 (RMS z)", variable=self.enable_artifact_filter
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(stats_frame, text="z阈值").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.artifact_z_entry = ttk.Entry(stats_frame, width=10)
        self.artifact_z_entry.insert(0, "3.5")
        self.artifact_z_entry.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        # 让右侧可以拉伸
        top.grid_columnconfigure(1, weight=1)

        # ========== 下部：图像区域（Notebook Tabs） ==========
        plot_area = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        plot_area.grid(row=1, column=0, sticky="nsew")

        self.nb = ttk.Notebook(plot_area)
        self.nb.pack(fill="both", expand=True)

        self.tab_single = ttk.Frame(self.nb)
        self.tab_multi = ttk.Frame(self.nb)
        self.tab_rms = ttk.Frame(self.nb)

        self.nb.add(self.tab_single, text="单通道 + 频段能量")
        self.nb.add(self.tab_multi, text="多通道叠加（论文级）")
        self.nb.add(self.tab_rms, text="RMS / ΔRMS")

        # --- 单通道：固定 2x2 轴（对比模式用满；非对比只用左列） ---
        self.fig_single = Figure(figsize=(9.2, 5.2), dpi=100)
        self.ax_wv_a = self.fig_single.add_subplot(2, 2, 1)   # waveform alert
        self.ax_wv_f = self.fig_single.add_subplot(2, 2, 2)   # waveform fatigue
        self.ax_bp_a = self.fig_single.add_subplot(2, 2, 3)   # bandpower alert
        self.ax_bp_f = self.fig_single.add_subplot(2, 2, 4)   # bandpower fatigue
        self._single_layout_positions = {
            "wv_a": self.ax_wv_a.get_position().frozen(),
            "wv_f": self.ax_wv_f.get_position().frozen(),
            "bp_a": self.ax_bp_a.get_position().frozen(),
            "bp_f": self.ax_bp_f.get_position().frozen(),
        }

        self.canvas_single = FigureCanvasTkAgg(self.fig_single, master=self.tab_single)
        self.canvas_single.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar_single = NavigationToolbar2Tk(self.canvas_single, self.tab_single)
        self.toolbar_single.update()

        # --- 多通道：1x2 轴（对比模式用两列；非对比只用左列） ---
        self.fig_multi = Figure(figsize=(9.2, 5.2), dpi=100)
        self.ax_multi_a = self.fig_multi.add_subplot(1, 2, 1)
        self.ax_multi_f = self.fig_multi.add_subplot(1, 2, 2)
        self._multi_layout_positions = {
            "multi_a": self.ax_multi_a.get_position().frozen(),
            "multi_f": self.ax_multi_f.get_position().frozen(),
        }

        self.canvas_multi = FigureCanvasTkAgg(self.fig_multi, master=self.tab_multi)
        self.canvas_multi.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar_multi = NavigationToolbar2Tk(self.canvas_multi, self.tab_multi)
        self.toolbar_multi.update()

        # --- RMS：固定主轴 + 固定颜色条轴（cax），避免反复挤压 ---
        self.fig_rms = Figure(figsize=(9.2, 5.0), dpi=100)
        self.ax_rms = self.fig_rms.add_subplot(111)
        # 固定颜色条轴（右侧预留）
        self.cax_rms = self.fig_rms.add_axes([0.90, 0.12, 0.02, 0.76])  # [left, bottom, width, height]
        self.im_rms = None
        self.cbar_rms = None

        self.canvas_rms = FigureCanvasTkAgg(self.fig_rms, master=self.tab_rms)
        self.canvas_rms.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar_rms = NavigationToolbar2Tk(self.canvas_rms, self.tab_rms)
        self.toolbar_rms.update()


        # ================== 脑区对比 Tab（RMS / 频段能量 / 差值） ==================
        self.tab_region = ttk.Frame(self.nb)
        self.nb.add(self.tab_region, text="脑区对比")

        # 脑区定义（A2 不参与）
        self.regions = {
            "前额区":  ["FP1","FP2","FZ","F3","F4","F7","F8","FCZ","FC3","FC4","FT7","FT8"],
            "中央区":  ["CZ","C3","C4"],
            "顶叶区": ["CPZ","CP3","CP4","PZ","P3","P4"],
            "颞叶区": ["T3","T4","T5","T6","TP7","TP8"],
            "枕叶区":["OZ","O1","O2"],
        }

        # 控件区
        ctrl = ttk.Frame(self.tab_region, padding=(10, 8, 10, 0))
        ctrl.pack(fill="x", expand=False)

        ttk.Label(ctrl, text="Feature").grid(row=0, column=0, sticky="w")
        self.region_feature_var = tk.StringVar(value="RMS")
        self.cb_region_feature = ttk.Combobox(
            ctrl,
            textvariable=self.region_feature_var,
            values=["RMS", "Band power", "Band ratio"],
            width=12,
            state="readonly",
        )
        self.cb_region_feature.grid(row=0, column=1, sticky="w", padx=(6, 14))

        ttk.Label(ctrl, text="Band").grid(row=0, column=2, sticky="w")
        self.region_band_var = tk.StringVar(value="theta")
        self.cb_region_band = ttk.Combobox(
            ctrl, textvariable=self.region_band_var, values=["delta", "theta", "alpha", "beta"], width=10, state="readonly"
        )
        self.cb_region_band.grid(row=0, column=3, sticky="w", padx=(6, 14))

        ttk.Label(ctrl, text="Ratio").grid(row=0, column=4, sticky="w")
        self.region_ratio_var = tk.StringVar(value="theta/alpha")
        self.cb_region_ratio = ttk.Combobox(
            ctrl,
            textvariable=self.region_ratio_var,
            values=["theta/alpha", "(theta+delta)/alpha"],
            width=16,
            state="readonly",
        )
        self.cb_region_ratio.grid(row=0, column=5, sticky="w", padx=(6, 14))

        ttk.Label(ctrl, text="Mode").grid(row=1, column=0, sticky="w")
        self.region_mode_var = tk.StringVar(value="Δ (Fatigue-Alert)")
        self.cb_region_mode = ttk.Combobox(
            ctrl,
            textvariable=self.region_mode_var,
            values=["EEG1 (Alert)", "EEG4 (Fatigue)", "Δ (Fatigue-Alert)"],
            width=16,
            state="readonly",
        )
        self.cb_region_mode.grid(row=1, column=1, sticky="w", padx=(6, 14))

        ttk.Label(ctrl, text="Agg").grid(row=1, column=2, sticky="w")
        self.region_agg_var = tk.StringVar(value="mean")
        self.cb_region_agg = ttk.Combobox(
            ctrl, textvariable=self.region_agg_var, values=["mean", "median"], width=8, state="readonly"
        )
        self.cb_region_agg.grid(row=1, column=3, sticky="w", padx=(6, 0))

        # Figure
        self.fig_region = Figure(figsize=(8.8, 4.6), dpi=100)
        self.ax_region = self.fig_region.add_subplot(111)
        self.canvas_region = FigureCanvasTkAgg(self.fig_region, master=self.tab_region)
        self.canvas_region.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar_region = NavigationToolbar2Tk(self.canvas_region, self.tab_region)
        self.toolbar_region.update()

        # 控件联动：任意变化都更新脑区图
        self.region_feature_var.trace_add("write", lambda *args: self.update_region_plot())
        self.region_band_var.trace_add("write", lambda *args: self.update_region_plot())
        self.region_ratio_var.trace_add("write", lambda *args: self.update_region_plot())
        self.region_mode_var.trace_add("write", lambda *args: self.update_region_plot())
        self.region_agg_var.trace_add("write", lambda *args: self.update_region_plot())

        # 缓存：RMS / Welch 能量矩阵（避免重复计算）
        self._metric_cache = {}  # key: (sid, sess) -> dict

        # 布局拉伸
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # 绑定：切换 RMS 模式时刷新（如果已加载过）
        self.rms_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_preview())
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._on_subject_select())
        self.compare_var.trace_add("write", lambda *_: self._on_mode_toggle())
        self.session_var.trace_add("write", lambda *_: self._maybe_preview())
        self.use_robust_epochs.trace_add("write", lambda *_: self._maybe_preview())
        self.enable_artifact_filter.trace_add("write", lambda *_: self._maybe_preview())

        self.status_var = tk.StringVar(value="就绪")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.grid(row=2, column=0, sticky="ew")

    # ---------- 数据加载 ----------
    def _load_eeg(self, sid: str, sess: str):
        key = (sid, sess)
        if key in self._cache:
            return self._cache[key]

        path = self.subjects.get(sid, {}).get(sess, None)
        if path is None:
            return None

        eeg = _safe_load_npy(path)
        self._cache[key] = eeg
        return eeg

    def _set_status(self, text: str):
        try:
            self.status_var.set(text)
        except Exception:
            pass

    def _maybe_preview(self):
        if self._has_loaded:
            self.on_preview()

    def _on_subject_select(self):
        idxs = self.listbox.curselection()
        if idxs:
            self._last_selected_index = int(idxs[0])
        self._maybe_preview()

    def _on_mode_toggle(self):
        if self.compare_var.get():
            try:
                self.rms_mode.set("ΔRMS(F-A)")
            except Exception:
                pass
        self._maybe_preview()

    def on_reset_controls(self):
        self.compare_var.set(False)
        self.session_var.set("EEG1")
        self.rms_mode.set("Alert")
        self.use_robust_epochs.set(True)
        self.enable_artifact_filter.set(True)
        self.epoch_entry.delete(0, tk.END)
        self.epoch_entry.insert(0, "0")
        self.channel_entry.delete(0, tk.END)
        self.channel_entry.insert(0, "0")
        self.epoch_pool_entry.delete(0, tk.END)
        self.epoch_pool_entry.insert(0, "20")
        self.artifact_z_entry.delete(0, tk.END)
        self.artifact_z_entry.insert(0, "3.5")
        self._set_status("设置已重置")

    def _parse_epoch_pool(self) -> int:
        try:
            n = int(self.epoch_pool_entry.get())
            return max(0, n)
        except Exception:
            return 0

    def _parse_artifact_z(self) -> float:
        try:
            return float(self.artifact_z_entry.get())
        except Exception:
            return 3.5

    def _epoch_selection_mask(self, rms_e: np.ndarray) -> np.ndarray:
        if rms_e is None or len(rms_e) == 0:
            return np.array([], dtype=bool)

        mask = np.ones(len(rms_e), dtype=bool)

        if self.enable_artifact_filter.get():
            z = _robust_zscore(rms_e)
            z_th = self._parse_artifact_z()
            mask &= np.abs(z) <= z_th

        if self.use_robust_epochs.get():
            n = self._parse_epoch_pool()
            if n > 0:
                valid = np.where(mask)[0]
                if len(valid) > n:
                    med = np.nanmedian(rms_e[valid])
                    order = np.argsort(np.abs(rms_e[valid] - med))
                    keep = valid[order[:n]]
                    new_mask = np.zeros_like(mask, dtype=bool)
                    new_mask[keep] = True
                    mask = new_mask

        return mask

    def _band_ratio_log(self, bp_lin: dict, ratio_name: str) -> np.ndarray:
        if bp_lin is None:
            return None
        if ratio_name == "theta/alpha":
            num = bp_lin.get("theta")
            den = bp_lin.get("alpha")
        elif ratio_name == "(theta+delta)/alpha":
            num = bp_lin.get("theta")
            den = bp_lin.get("alpha")
            if num is not None and bp_lin.get("delta") is not None:
                num = num + bp_lin.get("delta")
        else:
            return None
        if num is None or den is None:
            return None
        ratio = num / np.maximum(den, 1e-20)
        return np.log10(np.maximum(ratio, 1e-20))

    def _get_metrics(self, sid: str, sess: str):
        """
        返回该被试该 session 的缓存指标：
        - rms: (E,C)
        - bp_lin: dict {band: (E,C)} （Welch 线性域）
        - bp_log: dict {band: (E,C)} （log10 版本，更适合展示/对比）
        - ratio_log: dict {ratio: (E,C)}（log10 频段比值）
        """
        key = (sid, sess)
        if hasattr(self, "_metric_cache") and key in self._metric_cache:
            return self._metric_cache[key]

        eeg = self._load_eeg(sid, sess)
        if eeg is None:
            return None
        if eeg.ndim != 3:
            return None

        # RMS
        rms = np.sqrt(np.nanmean(eeg ** 2, axis=2))  # (E,C)

        # Welch 频段能量（线性域）
        bp_lin = _bandpower_welch_matrix(eeg, self.fs)

        # log10 版本：避免动态范围过大
        bp_log = {b: np.log10(np.maximum(mat, 1e-20)) for b, mat in bp_lin.items()}

        ratio_log = {
            "theta/alpha": self._band_ratio_log(bp_lin, "theta/alpha"),
            "(theta+delta)/alpha": self._band_ratio_log(bp_lin, "(theta+delta)/alpha"),
        }

        out = {"rms": rms, "bp_lin": bp_lin, "bp_log": bp_log, "ratio_log": ratio_log}

        if not hasattr(self, "_metric_cache"):
            self._metric_cache = {}
        self._metric_cache[key] = out
        return out

    def _region_aggregate(self, mat_ec: np.ndarray, agg: str = "mean", epoch_mask: np.ndarray = None):
        """
        将通道级矩阵聚合到脑区级（输出每个脑区一个标量）。
        输入：
        - mat_ec: shape (E,C) 或 (C,)
        规则：
        - 若有 epoch 维度：先对通道（脑区内）聚合 -> 得到 (E,)；再对 epoch 聚合 -> 标量
        - A2 不在 regions 中，因此不会参与
        """
        if mat_ec is None:
            return None

        if agg not in ("mean", "median"):
            agg = "mean"

        if mat_ec.ndim == 2 and epoch_mask is not None and len(epoch_mask) == mat_ec.shape[0]:
            mat_ec = mat_ec[epoch_mask]
        if mat_ec.ndim == 2 and mat_ec.shape[0] == 0:
            return {region: np.nan for region in self.regions.keys()}

        out = {}
        for region, chs in self.regions.items():
            idx = [self.channel_names.index(ch) for ch in chs if ch in self.channel_names]
            if not idx:
                out[region] = np.nan
                continue

            if mat_ec.ndim == 2:
                if agg == "mean":
                    v_e = np.nanmean(mat_ec[:, idx], axis=1)
                    out[region] = float(np.nanmean(v_e))
                else:
                    v_e = np.nanmedian(mat_ec[:, idx], axis=1)
                    out[region] = float(np.nanmedian(v_e))
            else:
                if agg == "mean":
                    out[region] = float(np.nanmean(mat_ec[idx]))
                else:
                    out[region] = float(np.nanmedian(mat_ec[idx]))

        return out

    def update_region_plot(self):
        """
        更新“脑区对比”Tab 的柱状图。
        - Feature: RMS / Band power（默认用 log10）
        - Band: delta/theta/alpha/beta（仅 Band power 有效）
        - Mode: EEG1 / EEG4 / Δ(F-A)
        - Agg: mean / median
        """
        # 只有在 tab 初始化完成后才更新
        if not hasattr(self, "ax_region"):
            return

        sid = self.get_selected_subject()
        if sid is None:
            return

        feature = self.region_feature_var.get()
        band = self.region_band_var.get()
        ratio = self.region_ratio_var.get()
        mode = self.region_mode_var.get()
        agg = self.region_agg_var.get()

        # Feature=RMS 时禁用 Band/Ratio 选择
        if feature == "RMS":
            try:
                self.cb_region_band.configure(state="disabled")
                self.cb_region_ratio.configure(state="disabled")
            except Exception:
                pass
        elif feature == "Band power":
            try:
                self.cb_region_band.configure(state="readonly")
                self.cb_region_ratio.configure(state="disabled")
            except Exception:
                pass
        else:
            try:
                self.cb_region_band.configure(state="disabled")
                self.cb_region_ratio.configure(state="readonly")
            except Exception:
                pass

        mat = None
        ylabel = ""
        title_mode = mode
        epoch_mask = None

        if mode == "EEG1 (Alert)":
            m1 = self._get_metrics(sid, "EEG1")
            if m1 is None:
                return
            if feature == "RMS":
                mat = m1["rms"]
                ylabel = "RMS (a.u.)"
                epoch_mask = self._epoch_selection_mask(np.nanmean(m1["rms"], axis=1))
            elif feature == "Band power":
                mat = m1["bp_log"][band]
                ylabel = f"log10 band power ({band})"
                epoch_mask = self._epoch_selection_mask(np.nanmean(m1["rms"], axis=1))
            else:
                mat = m1["ratio_log"][ratio]
                ylabel = f"log10 ratio ({ratio})"
                epoch_mask = self._epoch_selection_mask(np.nanmean(m1["rms"], axis=1))
            bar_color = ALERT_COLOR  # EEG1：粉色
        elif mode == "EEG4 (Fatigue)":
            m4 = self._get_metrics(sid, "EEG4")
            if m4 is None:
                return
            if feature == "RMS":
                mat = m4["rms"]
                ylabel = "RMS (a.u.)"
                epoch_mask = self._epoch_selection_mask(np.nanmean(m4["rms"], axis=1))
            elif feature == "Band power":
                mat = m4["bp_log"][band]
                ylabel = f"log10 band power ({band})"
                epoch_mask = self._epoch_selection_mask(np.nanmean(m4["rms"], axis=1))
            else:
                mat = m4["ratio_log"][ratio]
                ylabel = f"log10 ratio ({ratio})"
                epoch_mask = self._epoch_selection_mask(np.nanmean(m4["rms"], axis=1))
            bar_color = FATIGUE_COLOR  # EEG4：蓝色
        else:
            m1 = self._get_metrics(sid, "EEG1")
            m4 = self._get_metrics(sid, "EEG4")
            if (m1 is None) or (m4 is None):
                return
            if feature == "RMS":
                E = min(m1["rms"].shape[0], m4["rms"].shape[0])
                mat = m4["rms"][:E] - m1["rms"][:E]
                ylabel = "ΔRMS (Fatigue − Alert)"
                rms_e = 0.5 * (np.nanmean(m1["rms"], axis=1)[:E] + np.nanmean(m4["rms"], axis=1)[:E])
                epoch_mask = self._epoch_selection_mask(rms_e)
            elif feature == "Band power":
                E = min(m1["bp_log"][band].shape[0], m4["bp_log"][band].shape[0])
                mat = m4["bp_log"][band][:E] - m1["bp_log"][band][:E]  # log 域差值 = log ratio
                ylabel = f"Δ log10 power ({band})"
                rms_e = 0.5 * (np.nanmean(m1["rms"], axis=1)[:E] + np.nanmean(m4["rms"], axis=1)[:E])
                epoch_mask = self._epoch_selection_mask(rms_e)
            else:
                E = min(m1["ratio_log"][ratio].shape[0], m4["ratio_log"][ratio].shape[0])
                mat = m4["ratio_log"][ratio][:E] - m1["ratio_log"][ratio][:E]
                ylabel = f"Δ log10 ratio ({ratio})"
                rms_e = 0.5 * (np.nanmean(m1["rms"], axis=1)[:E] + np.nanmean(m4["rms"], axis=1)[:E])
                epoch_mask = self._epoch_selection_mask(rms_e)
            bar_color = None  # Δ：按正负着色

        if epoch_mask is not None and mat is not None and mat.ndim == 2:
            E = min(mat.shape[0], len(epoch_mask))
            mat = mat[:E]
            epoch_mask = epoch_mask[:E]

        region_vals = self._region_aggregate(mat, agg=agg, epoch_mask=epoch_mask)
        if region_vals is None:
            return

        regions = list(region_vals.keys())
        vals = [region_vals[r] for r in regions]

        if bar_color is not None:
            colors = [bar_color] * len(vals)
        else:
            colors = [DELTA_POS_COLOR if (not np.isnan(v) and v >= 0) else DELTA_NEG_COLOR for v in vals]

        self.ax_region.clear()
        self.ax_region.bar(regions, vals, color=colors, edgecolor="black", linewidth=0.6)
        self.ax_region.set_title(f"Region-level comparison | {sid} | {title_mode}")
        self.ax_region.set_ylabel(ylabel)
        self.ax_region.spines["top"].set_visible(False)
        self.ax_region.spines["right"].set_visible(False)
        self.ax_region.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.35)

        self.fig_region.tight_layout()
        self.canvas_region.draw()


    def get_selected_subject(self):
        idxs = self.listbox.curselection()
        if not idxs:
            if self.sub_ids:
                idx = min(max(self._last_selected_index, 0), len(self.sub_ids) - 1)
                self.listbox.selection_set(idx)
                self.listbox.activate(idx)
                self._last_selected_index = idx
                return self.sub_ids[idx]
            return None
        idx = int(idxs[0])
        self._last_selected_index = idx
        return self.sub_ids[idx]

    def on_show_path(self):
        sid = self.get_selected_subject()
        if sid is None:
            messagebox.showwarning("提示", "请先选择一个被试。")
            return

        if self.compare_var.get():
            p1 = self.subjects.get(sid, {}).get("EEG1", None)
            p4 = self.subjects.get(sid, {}).get("EEG4", None)
            messagebox.showinfo("文件路径", f"EEG1:\n{p1}\n\nEEG4:\n{p4}")
            return

        sess = self.session_var.get().upper()
        path = self.subjects.get(sid, {}).get(sess, None)
        if path is None:
            messagebox.showwarning("提示", f"该被试缺少 {sess} 文件。")
            return
        messagebox.showinfo("文件路径", path)

    def _validate_shape(self, eeg: np.ndarray, sid: str, sess: str):
        if eeg is None:
            raise ValueError(f"{sid} 缺少 {sess} 文件")
        if eeg.ndim != 3:
            raise ValueError(f"{sid} {sess} 仅支持 3D (E,C,T)，但 shape={eeg.shape}")
        E, C, T = eeg.shape
        if E <= 0 or C <= 0 or T <= 0:
            raise ValueError(f"{sid} {sess} shape 非法：{eeg.shape}")
        # 通道名长度检查（不强制报错：允许你未来换 C）
        if len(self.channel_names) != C:
            # 只提示，不中断
            print(f"[警告] 通道名数量={len(self.channel_names)} 与 C={C} 不一致，将用索引标签。")
        return E, C, T

    # ---------- 功能 3：代表性 epoch ----------
    def on_pick_representative_epoch(self):
        sid = self.get_selected_subject()
        if sid is None:
            messagebox.showwarning("提示", "请先选择一个被试。")
            return
        try:
            if self.compare_var.get():
                eeg1 = self._load_eeg(sid, "EEG1")
                eeg4 = self._load_eeg(sid, "EEG4")
                if eeg1 is None or eeg4 is None:
                    raise ValueError("对比模式需要同时存在 EEG1 与 EEG4。")

                # 代表性 epoch：对两个状态分别取 RMS 中位数 idx，然后取其平均更稳
                rms1 = _epoch_rms_vector(eeg1)
                rms4 = _epoch_rms_vector(eeg4)
                mask1 = self._epoch_selection_mask(rms1)
                mask4 = self._epoch_selection_mask(rms4)
                idx1 = _representative_epoch_idx_with_mask(eeg1, mask1)
                idx4 = _representative_epoch_idx_with_mask(eeg4, mask4)
                rep = int(round((idx1 + idx4) / 2))
            else:
                sess = self.session_var.get().upper()
                eeg = self._load_eeg(sid, sess)
                if eeg is None:
                    raise ValueError(f"缺少 {sess}")
                rms = _epoch_rms_vector(eeg)
                mask = self._epoch_selection_mask(rms)
                rep = _representative_epoch_idx_with_mask(eeg, mask)

            self.epoch_entry.delete(0, tk.END)
            self.epoch_entry.insert(0, str(rep))
            self._set_status(f"代表性 epoch 已更新：{rep}")
            self._maybe_preview()
        except Exception as e:
            messagebox.showerror("错误", f"代表性 epoch 计算失败：\n{e}")

    # ---------- 主刷新 ----------
    def on_preview(self):
        sid = self.get_selected_subject()
        if sid is None:
            messagebox.showwarning("提示", "请先选择一个被试。")
            return

        # 读取 epoch/channel
        try:
            epoch_idx = int(self.epoch_entry.get())
            channel_idx = int(self.channel_entry.get())
        except Exception:
            messagebox.showerror("错误", "Epoch/Channel 必须是整数。")
            return

        compare = bool(self.compare_var.get())

        try:
            if compare:
                eeg1 = self._load_eeg(sid, "EEG1")
                eeg4 = self._load_eeg(sid, "EEG4")
                if eeg1 is None or eeg4 is None:
                    raise ValueError("对比模式需要同时存在 EEG1 与 EEG4。")

                E1, C1, T1 = self._validate_shape(eeg1, sid, "EEG1")
                E4, C4, T4 = self._validate_shape(eeg4, sid, "EEG4")

                # 允许 E 不同：取各自范围内 clip
                epoch_a = max(0, min(epoch_idx, E1 - 1))
                epoch_f = max(0, min(epoch_idx, E4 - 1))
                ch_a = max(0, min(channel_idx, C1 - 1))
                ch_f = max(0, min(channel_idx, C4 - 1))

                # --- 1) 单通道波形 + 频段能量（并排） ---
                self._update_single_layout(compare=True)
                self._draw_single_and_bandpower_compare(sid, epoch_a, ch_a, eeg1, "EEG1",
                                                       epoch_f, ch_f, eeg4, "EEG4")

                # --- 2) 多通道叠加（论文级，并排） ---
                self._update_multi_layout(compare=True)
                self._draw_stacked_compare(sid, epoch_a, eeg1, "EEG1", epoch_f, eeg4, "EEG4")

                # --- 3) RMS / ΔRMS ---
                self._draw_rms_mode(sid, eeg1, eeg4)

                # 更新脑区对比 Tab
                self.update_region_plot()

            else:
                sess = self.session_var.get().upper()
                eeg = self._load_eeg(sid, sess)
                if eeg is None:
                    raise ValueError(f"该被试缺少 {sess} 文件。")

                E, C, T = self._validate_shape(eeg, sid, sess)
                epoch_idx = max(0, min(epoch_idx, E - 1))
                channel_idx = max(0, min(channel_idx, C - 1))

                # 单通道 + 频段能量：仅使用左列，右列隐藏
                self._update_single_layout(compare=False)
                self._draw_single_and_bandpower_single(sid, sess, eeg, epoch_idx, channel_idx)

                # 多通道叠加：仅左列绘制，右列隐藏
                self._update_multi_layout(compare=False)
                self._draw_stacked_single(sid, sess, eeg, epoch_idx)

                # RMS：按下拉框显示（Alert/Fatigue/ΔRMS）；单模式下 ΔRMS 将提示
                self._draw_rms_single_mode(sid, sess, eeg)

            # 更新脑区对比 Tab
            self.update_region_plot()
            self._has_loaded = True
            if compare:
                self.data_info_label.config(text=f"EEG1: E={E1} C={C1} T={T1} | EEG4: E={E4} C={C4} T={T4}")
                self._set_status(f"已加载 {sid} | 对比模式 | Epoch {epoch_idx} | Ch {channel_idx}")
            else:
                self.data_info_label.config(text=f"{sess}: E={E} C={C} T={T}")
                self._set_status(f"已加载 {sid} | {sess} | Epoch {epoch_idx} | Ch {channel_idx}")

        except Exception as e:
            messagebox.showerror("错误", f"加载/绘图失败：\n{e}")
            self._set_status(f"加载失败：{e}")

    # ---------- 绘图：单通道 + 频段能量 ----------
    def _hide_axis(self, ax):
        ax.clear()
        ax.set_axis_off()

    def _get_channel_labels(self, C: int):
        if len(self.channel_names) == C:
            return self.channel_names
        return [f"Ch{c}" for c in range(C)]

    def _draw_single_and_bandpower_single(self, sid, sess, eeg, epoch_idx, channel_idx):
        E, C, T = eeg.shape
        labels = self._get_channel_labels(C)
        ch_name = labels[channel_idx]
        t = np.arange(T) / self.fs

        # 左上：波形
        self.ax_wv_a.clear()
        sig = eeg[epoch_idx, channel_idx, :]
        color = ALERT_COLOR if sess.upper() == "EEG1" else FATIGUE_COLOR
        self.ax_wv_a.plot(t, sig, linewidth=1.0, color=color)
        self.ax_wv_a.set_title(f"Waveform | {sid} | {sess} | Epoch {epoch_idx} | {ch_name}")
        self.ax_wv_a.set_xlabel("Time (s)")
        self.ax_wv_a.set_ylabel("Amplitude")
        self.ax_wv_a.spines["top"].set_visible(False)
        self.ax_wv_a.spines["right"].set_visible(False)

        # 右上、右下：隐藏
        self._hide_axis(self.ax_wv_f)
        self._hide_axis(self.ax_bp_f)

        # 左下：频段能量条形图
        self.ax_bp_a.clear()
        bp = _bandpowers_fft(sig, self.fs)
        bands = ["delta", "theta", "alpha", "beta"]
        vals = [bp[b] for b in bands]
        self.ax_bp_a.bar(bands, vals, color=color)
        self.ax_bp_a.set_title("Band power (FFT) | δ θ α β")
        self.ax_bp_a.set_ylabel("Power (a.u.)")
        self.ax_bp_a.spines["top"].set_visible(False)
        self.ax_bp_a.spines["right"].set_visible(False)

        self.fig_single.tight_layout()
        self.canvas_single.draw()

    def _draw_single_and_bandpower_compare(self, sid, epoch_a, ch_a, eeg1, sess1, epoch_f, ch_f, eeg4, sess4):
        E1, C1, T1 = eeg1.shape
        E4, C4, T4 = eeg4.shape

        labels1 = self._get_channel_labels(C1)
        labels4 = self._get_channel_labels(C4)

        t1 = np.arange(T1) / self.fs
        t4 = np.arange(T4) / self.fs

        sig1 = eeg1[epoch_a, ch_a, :]
        sig4 = eeg4[epoch_f, ch_f, :]

        # 左上：EEG1 波形
        self.ax_wv_a.clear()
        self.ax_wv_a.plot(t1, sig1, linewidth=1.0, color=ALERT_COLOR)
        self.ax_wv_a.set_title(f"Waveform | {sid} | {sess1} | Epoch {epoch_a} | {labels1[ch_a]}")
        self.ax_wv_a.set_xlabel("Time (s)")
        self.ax_wv_a.set_ylabel("Amplitude")
        self.ax_wv_a.spines["top"].set_visible(False)
        self.ax_wv_a.spines["right"].set_visible(False)

        # 右上：EEG4 波形
        self.ax_wv_f.clear()
        self.ax_wv_f.plot(t4, sig4, linewidth=1.0, color=FATIGUE_COLOR)
        self.ax_wv_f.set_title(f"Waveform | {sid} | {sess4} | Epoch {epoch_f} | {labels4[ch_f]}")
        self.ax_wv_f.set_xlabel("Time (s)")
        self.ax_wv_f.set_ylabel("Amplitude")
        self.ax_wv_f.spines["top"].set_visible(False)
        self.ax_wv_f.spines["right"].set_visible(False)

        # 左下：EEG1 频段能量
        self.ax_bp_a.clear()
        bp1 = _bandpowers_fft(sig1, self.fs)
        bands = ["delta", "theta", "alpha", "beta"]
        self.ax_bp_a.bar(bands, [bp1[b] for b in bands], color=ALERT_COLOR)
        self.ax_bp_a.set_title("Band power | EEG1")
        self.ax_bp_a.set_ylabel("Power (a.u.)")
        self.ax_bp_a.spines["top"].set_visible(False)
        self.ax_bp_a.spines["right"].set_visible(False)

        # 右下：EEG4 频段能量
        self.ax_bp_f.clear()
        bp4 = _bandpowers_fft(sig4, self.fs)
        self.ax_bp_f.bar(bands, [bp4[b] for b in bands], color=FATIGUE_COLOR)
        self.ax_bp_f.set_title("Band power | EEG4")
        self.ax_bp_f.set_ylabel("Power (a.u.)")
        self.ax_bp_f.spines["top"].set_visible(False)
        self.ax_bp_f.spines["right"].set_visible(False)

        self.fig_single.tight_layout()
        self.canvas_single.draw()

    # ---------- 绘图：多通道叠加 ----------
    def _draw_stacked_single(self, sid, sess, eeg, epoch_idx):
        E, C, T = eeg.shape
        labels = self._get_channel_labels(C)

        epoch_data = eeg[epoch_idx, :, :]
        plot_stacked_paper_on_ax(
            ax=self.ax_multi_a,
            eeg_epoch=epoch_data,
            fs=self.fs,
            channel_names=labels,
            title=f"Stacked EEG (paper-style) | {sid} | {sess} | Epoch {epoch_idx}",
            mode="robust_z",
            linewidth=0.7,
        )
        # 右列隐藏
        self._hide_axis(self.ax_multi_f)

        self.fig_multi.tight_layout()
        self.canvas_multi.draw()

    def _draw_stacked_compare(self, sid, epoch_a, eeg1, sess1, epoch_f, eeg4, sess4):
        E1, C1, T1 = eeg1.shape
        E4, C4, T4 = eeg4.shape
        labels1 = self._get_channel_labels(C1)
        labels4 = self._get_channel_labels(C4)

        plot_stacked_paper_on_ax(
            ax=self.ax_multi_a,
            eeg_epoch=eeg1[epoch_a, :, :],
            fs=self.fs,
            channel_names=labels1,
            title=f"Stacked | {sid} | {sess1} | Epoch {epoch_a}",
            mode="robust_z",
            linewidth=0.7,
        )
        plot_stacked_paper_on_ax(
            ax=self.ax_multi_f,
            eeg_epoch=eeg4[epoch_f, :, :],
            fs=self.fs,
            channel_names=labels4,
            title=f"Stacked | {sid} | {sess4} | Epoch {epoch_f}",
            mode="robust_z",
            linewidth=0.7,
        )

        self.fig_multi.tight_layout()
        self.canvas_multi.draw()

    def _update_single_layout(self, compare: bool):
        if compare:
            self.ax_wv_a.set_position(self._single_layout_positions["wv_a"])
            self.ax_wv_f.set_position(self._single_layout_positions["wv_f"])
            self.ax_bp_a.set_position(self._single_layout_positions["bp_a"])
            self.ax_bp_f.set_position(self._single_layout_positions["bp_f"])
            self.ax_wv_f.set_visible(True)
            self.ax_bp_f.set_visible(True)
            self.ax_wv_f.set_axis_on()
            self.ax_bp_f.set_axis_on()
        else:
            self.ax_wv_a.set_position([0.08, 0.55, 0.86, 0.35])
            self.ax_bp_a.set_position([0.08, 0.10, 0.86, 0.35])
            self.ax_wv_f.set_visible(False)
            self.ax_bp_f.set_visible(False)
        self.canvas_single.draw()

    def _update_multi_layout(self, compare: bool):
        if compare:
            self.ax_multi_a.set_position(self._multi_layout_positions["multi_a"])
            self.ax_multi_f.set_position(self._multi_layout_positions["multi_f"])
            self.ax_multi_f.set_visible(True)
            self.ax_multi_f.set_axis_on()
        else:
            self.ax_multi_a.set_position([0.08, 0.12, 0.86, 0.78])
            self.ax_multi_f.set_visible(False)
        self.canvas_multi.draw()

    # ---------- 绘图：RMS / ΔRMS ----------
    def _ensure_rms_canvas(self, im):
        """首次创建 colorbar；后续只更新。"""
        if self.cbar_rms is None:
            self.cbar_rms = self.fig_rms.colorbar(im, cax=self.cax_rms)
            self.cbar_rms.set_label("RMS")

    def _draw_rms_mode(self, sid, eeg1, eeg4):
        """
        对比模式下：根据下拉框显示 Alert / Fatigue / ΔRMS
        """
        mode = self.rms_mode.get()
        rms1 = np.sqrt(np.nanmean(eeg1 ** 2, axis=2))  # (E,C)
        rms4 = np.sqrt(np.nanmean(eeg4 ** 2, axis=2))  # (E,C)

        if mode == "Alert":
            mat = rms1
            title = f"RMS heatmap | {sid} | EEG1 (Alert)"
            cbar_label = "RMS"
        elif mode == "Fatigue":
            mat = rms4
            title = f"RMS heatmap | {sid} | EEG4 (Fatigue)"
            cbar_label = "RMS"
        else:
            # ΔRMS(F-A)
            # 允许 E/C 不一致：取交集尺寸
            E = min(rms1.shape[0], rms4.shape[0])
            C = min(rms1.shape[1], rms4.shape[1])
            mat = rms4[:E, :C] - rms1[:E, :C]
            title = f"ΔRMS (Fatigue - Alert) | {sid}"
            cbar_label = "ΔRMS"

        if self.im_rms is None:
            self.ax_rms.clear()
            self.im_rms = self.ax_rms.imshow(mat, aspect="auto", origin="lower")
            self.ax_rms.set_title(title)
            self.ax_rms.set_xlabel("Channel")
            self.ax_rms.set_ylabel("Epoch")
            self._ensure_rms_canvas(self.im_rms)
            self.cbar_rms.set_label(cbar_label)
        else:
            self.im_rms.set_data(mat)
            self.ax_rms.set_title(title)
            self.im_rms.set_clim(vmin=np.nanmin(mat), vmax=np.nanmax(mat))
            self.cbar_rms.update_normal(self.im_rms)
            self.cbar_rms.set_label(cbar_label)

        # 固定边距（避免挤压）
        self.fig_rms.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.90)
        self.canvas_rms.draw()

    def _draw_rms_single_mode(self, sid, sess, eeg):
        mode = self.rms_mode.get()
        if mode == "ΔRMS(F-A)":
            # 单模式下没有对比数据
            self.ax_rms.clear()
            self.ax_rms.text(
                0.5, 0.5,
                "ΔRMS 需要开启对比模式\n(同时加载 EEG1 与 EEG4)",
                ha="center", va="center", transform=self.ax_rms.transAxes
            )
            self.ax_rms.set_axis_off()
            self.canvas_rms.draw()
            return

        rms = np.sqrt(np.nanmean(eeg ** 2, axis=2))  # (E,C)
        title = f"RMS heatmap | {sid} | {sess}"

        if self.im_rms is None:
            self.ax_rms.clear()
            self.im_rms = self.ax_rms.imshow(rms, aspect="auto", origin="lower")
            self.ax_rms.set_title(title)
            self.ax_rms.set_xlabel("Channel")
            self.ax_rms.set_ylabel("Epoch")
            self._ensure_rms_canvas(self.im_rms)
            self.cbar_rms.set_label("RMS")
        else:
            self.im_rms.set_data(rms)
            self.ax_rms.set_title(title)
            self.im_rms.set_clim(vmin=np.nanmin(rms), vmax=np.nanmax(rms))
            self.cbar_rms.update_normal(self.im_rms)
            self.cbar_rms.set_label("RMS")

        self.fig_rms.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.90)
        self.canvas_rms.draw()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ROOT_DIR = r"E:\TJU_File\PythonProject2\Data\Datafor10kids S1&S4 6mins"
    FS = 1000
    app = EEGGuiBrowserEmbed(ROOT_DIR, fs=FS)
    app.run()
