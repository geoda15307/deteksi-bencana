"""
GUI Klasifikasi Citra Bencana — MobileNetV2
===========================================
Jalankan: streamlit run app.py
"""

import json, os, time
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DetekSi Bencana",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { background: #eef5ff !important; }
html { scroll-behavior: smooth; }
body, .stApp {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
div[data-testid="stAppViewBlockContainer"] { padding: 0 !important; }

/* ── Typography ── */
* { font-family: 'Inter', sans-serif !important; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* Pengecualian penting: ikon bawaan Streamlit (mis. ikon "upload" pada
   tombol file uploader) sebenarnya adalah font ikon (Material Symbols),
   bukan gambar. Kalau ikut dipaksa pakai font Inter, kode ikonnya
   ("upload") akan tampil sebagai TEKS biasa alih-alih simbol — itulah
   penyebab tulisan "uploadUpload" yang tampak dobel/tumpang tindih. */
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    font-size: 1.1rem !important;
}

/* ── Navbar ── */
.navbar {
    background: rgba(255, 255, 255, 0.96);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-bottom: 1px solid rgba(37, 99, 235, 0.11);
    padding: 0 2.5rem;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 2px 16px rgba(37, 99, 235, 0.07);
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #1e3a5f;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: -0.02em;
}
.nav-logo .dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #ef4444;
    animation: pulse 2s infinite;
    box-shadow: 0 0 0 3px rgba(239,68,68,0.2);
}
.nav-badge {
    background: linear-gradient(135deg, rgba(37,99,235,0.1), rgba(96,165,250,0.15));
    border: 1px solid rgba(37,99,235,0.25);
    color: #2563eb;
    font-size: 0.66rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 20px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace !important;
}
.nav-powered {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 500;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 3px rgba(239,68,68,0.2); }
    50% { opacity: 0.7; transform: scale(0.85); box-shadow: 0 0 0 8px rgba(239,68,68,0.04); }
}

/* ══════════════════════════════════════════════════
   ANIMATION KEYFRAMES
══════════════════════════════════════════════════ */

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.88); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-28px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(28px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes barGrow {
    from { width: 0% !important; }
    to   { width: var(--bar-w) !important; }
}
@keyframes float {
    0%, 100% { transform: translate(-50%, -28%) scale(1); opacity: 0.9; }
    50%       { transform: translate(-50%, -30%) scale(1.02); opacity: 0.6; }
}
@keyframes floatSlow {
    0%, 100% { transform: translate(-50%, -28%) scale(1); }
    50%       { transform: translate(-50%, -26%) scale(1.015); }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes badgeFloat {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-3px); }
}
@keyframes ringRotate {
    from { transform: translate(-50%, -28%) rotate(0deg); }
    to   { transform: translate(-50%, -28%) rotate(360deg); }
}
@keyframes gradShift {
    0%, 100% { background-position: 0% 50%; }
    50%       { background-position: 100% 50%; }
}
@keyframes popIn {
    0%   { opacity: 0; transform: scale(0.7) translateY(10px); }
    70%  { transform: scale(1.05) translateY(-2px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

/* ══════════════════════════════════════════════════
   APPLY ANIMATIONS — GLOBAL ELEMENTS
══════════════════════════════════════════════════ */

/* Navbar — slide in dari atas */
.navbar {
    animation: fadeInDown 0.55s cubic-bezier(0.22,1,0.36,1) both;
}

/* Hero elements — staggered */
.hero-eyebrow {
    animation: fadeInDown 0.5s 0.1s cubic-bezier(0.22,1,0.36,1) both;
}
.hero h1 {
    animation: fadeInUp 0.6s 0.2s cubic-bezier(0.22,1,0.36,1) both;
}
.hero p {
    animation: fadeInUp 0.6s 0.35s cubic-bezier(0.22,1,0.36,1) both;
}

/* Hero rings — floating */
.hero::before {
    animation: float 7s ease-in-out infinite;
}
.hero::after {
    animation: float 9s 1s ease-in-out infinite;
}
.hero-ring-inner {
    animation: floatSlow 5s 0.5s ease-in-out infinite;
}

/* Stats bar — staggered fade in */
.stat-item:nth-child(1) { animation: fadeInUp 0.5s 0.3s both; }
.stat-item:nth-child(2) { animation: fadeInUp 0.5s 0.42s both; }
.stat-item:nth-child(3) { animation: fadeInUp 0.5s 0.54s both; }
.stat-item:nth-child(4) { animation: fadeInUp 0.5s 0.66s both; }

/* Stat value — gradient animation */
.stat-value {
    background-size: 200% 200%;
    animation: gradShift 4s ease infinite;
}

/* Panel icon badge — float */
.panel-icon-badge {
    animation: badgeFloat 3s ease-in-out infinite;
}

/* Cards — entrance */
div[data-testid="stVerticalBlockBorderWrapper"] {
    animation: fadeInUp 0.55s cubic-bezier(0.22,1,0.36,1) both;
    transition: box-shadow 0.3s ease, transform 0.25s cubic-bezier(0.22,1,0.36,1),
                border-color 0.25s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(37,99,235,0.18),
                0 2px 8px rgba(37,99,235,0.1) !important;
    border-color: rgba(37,99,235,0.42) !important;
}

/* Pred header — pop in */
.pred-header {
    animation: popIn 0.5s cubic-bezier(0.22,1,0.36,1) both;
}

/* Confidence badge — scale in */
.pred-conf, .pred-conf-warn, .pred-conf-low {
    animation: scaleIn 0.4s 0.2s cubic-bezier(0.22,1,0.36,1) both;
}

/* Bar rows — staggered slide in */
.bar-row:nth-child(1) { animation: slideInLeft 0.45s 0.05s both; }
.bar-row:nth-child(2) { animation: slideInLeft 0.45s 0.15s both; }
.bar-row:nth-child(3) { animation: slideInLeft 0.45s 0.25s both; }
.bar-row:nth-child(4) { animation: slideInLeft 0.45s 0.35s both; }
.bar-row:nth-child(5) { animation: slideInLeft 0.45s 0.45s both; }
.bar-row:nth-child(6) { animation: slideInLeft 0.45s 0.55s both; }

/* Grad-CAM section — fade in up */
.st-key-gradcam_section {
    animation: fadeInUp 0.6s 0.1s cubic-bezier(0.22,1,0.36,1) both;
}

/* Expander — fade in */
div[data-testid="stExpander"] {
    animation: fadeInUp 0.5s 0.15s both;
    transition: box-shadow 0.3s ease !important;
}

/* Analyse button — enhanced transitions */
.stButton > button {
    transition: all 0.28s cubic-bezier(0.22,1,0.36,1) !important;
    position: relative !important;
    overflow: hidden !important;
}
/* Ripple shimmer on button hover */
.stButton > button::after {
    content: '' !important;
    position: absolute !important;
    top: 0 !important; left: -100% !important;
    width: 60% !important; height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent) !important;
    transition: left 0.5s ease !important;
}
.stButton > button:hover::after {
    left: 160% !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.012) !important;
    box-shadow: 0 12px 36px rgba(37,99,235,0.48),
                0 1px 0 rgba(255,255,255,0.15) inset !important;
}
.stButton > button:active {
    transform: translateY(-1px) scale(0.99) !important;
    transition-duration: 0.1s !important;
}

/* Dropzone — smooth hover */
.dropzone {
    transition: all 0.3s cubic-bezier(0.22,1,0.36,1) !important;
}
.dropzone:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(37,99,235,0.12);
}

/* Img meta — slide up */
.img-meta {
    animation: fadeInUp 0.35s 0.1s both;
}

/* Result empty — fade in */
.result-empty {
    animation: fadeIn 0.5s both;
}

/* Gradcam info — fade in */
.gradcam-info {
    animation: fadeInUp 0.4s 0.2s both;
}
.gradcam-note {
    animation: fadeInUp 0.4s 0.3s both;
}

/* About grid items — stagger */
.about-item {
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.about-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(37,99,235,0.12);
}
.about-thresh-row {
    transition: all 0.2s ease;
}
.about-thresh-row:hover {
    background: #eff6ff !important;
    transform: translateX(4px);
    border-color: rgba(37,99,235,0.22) !important;
}

/* Footer — fade in */
.footer {
    animation: fadeIn 0.6s 0.4s both;
}

/* Scrollbar — custom smooth */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #eef5ff; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #3b82f6, #60a5fa);
    border-radius: 6px;
}
::-webkit-scrollbar-thumb:hover { background: #2563eb; }


/* ── Hero ── */
.hero {
    padding: 4.5rem 2rem 3.5rem;
    text-align: center;
    background: linear-gradient(180deg, #dbeafe 0%, #eff6ff 55%, #eef5ff 100%);
    border-bottom: 1px solid rgba(37, 99, 235, 0.1);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    width: 620px; height: 620px;
    border-radius: 50%;
    border: 1px solid rgba(37,99,235,0.09);
    top: 50%; left: 50%;
    transform: translate(-50%, -28%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    width: 390px; height: 390px;
    border-radius: 50%;
    border: 1px solid rgba(96,165,250,0.13);
    top: 50%; left: 50%;
    transform: translate(-50%, -28%);
    pointer-events: none;
}
.hero-ring-inner {
    position: absolute;
    width: 200px; height: 200px;
    border-radius: 50%;
    border: 1px solid rgba(37,99,235,0.16);
    top: 50%; left: 50%;
    transform: translate(-50%, -28%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.2);
    color: #2563eb;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 1.4rem;
    font-family: 'JetBrains Mono', monospace !important;
    position: relative;
}
.hero h1 {
    font-size: clamp(2rem, 4.5vw, 3.4rem);
    font-weight: 800;
    color: #1e3a5f;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 1.15rem;
    position: relative;
}
.hero h1 span {
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #64748b;
    font-size: 0.96rem;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.85;
    position: relative;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 0;
    background: #ffffff;
    border-bottom: 1px solid rgba(37,99,235,0.09);
    flex-wrap: wrap;
    box-shadow: 0 2px 10px rgba(37,99,235,0.05);
}
.stat-item {
    text-align: center;
    padding: 1.35rem 3rem;
    position: relative;
}
.stat-item::after {
    content: '';
    position: absolute;
    right: 0; top: 25%; bottom: 25%;
    width: 1px;
    background: rgba(37,99,235,0.1);
}
.stat-item:last-child::after { display: none; }
.stat-value {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.45rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
}
.stat-label {
    font-size: 0.63rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    margin-top: 4px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace !important;
}
.stat-divider { display: none; }

/* ── Main card area — dibungkus st.container(key="main_area") asli,
       supaya lebar maksimum & padding samping benar-benar berlaku
       (kotak kiri-kanan tidak lagi mentok ke tepi layar). ── */
.st-key-main_area {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2.25rem 2rem 4rem;
}

/* ── Panel header strip ── */
.panel-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(37,99,235,0.08);
}
.panel-header-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2563eb, #60a5fa);
    flex-shrink: 0;
}
.panel-label {
    font-size: 0.65rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Panel header utama (judul besar dengan badge ikon) ── */
.panel-header-main {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1.15rem;
    border-bottom: 1px solid rgba(37,99,235,0.08);
}
.panel-icon-badge {
    width: 42px; height: 42px;
    flex-shrink: 0;
    border-radius: 13px;
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #60a5fa 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 6px 16px rgba(37,99,235,0.32), 0 1px 0 rgba(255,255,255,0.25) inset;
}
.panel-title-group {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.panel-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e3a5f;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.panel-subtitle {
    font-size: 0.74rem;
    color: #94a3b8;
    font-weight: 500;
}

/* ── Panel (kartu utama) — dibuat dari st.container(border=True) ──
       Latar diberi warna sedikit gelap (bukan putih polos) supaya
       judul & isi di dalamnya terasa "menyatu" dengan kartunya. ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    position: relative;
    background: #e7eefc !important;
    border-radius: 20px !important;
    padding: 1.75rem !important;
    box-shadow: 0 4px 22px rgba(37, 99, 235, 0.12),
                0 1px 4px rgba(37, 99, 235, 0.08) !important;
    border: 1.5px solid rgba(37, 99, 235, 0.28) !important;
    transition: box-shadow 0.3s ease, transform 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 8px 34px rgba(37, 99, 235, 0.16),
                0 2px 8px rgba(37, 99, 235, 0.1) !important;
}

/* ── Drop zone ── */
.dropzone {
    border: 1.5px dashed rgba(37,99,235,0.25);
    border-radius: 16px;
    padding: 2.75rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s;
    background: rgba(37,99,235,0.03);
    margin-top: 0.5rem;
}
.dropzone:hover {
    border-color: rgba(37,99,235,0.48);
    background: rgba(37,99,235,0.06);
}
.dropzone-icon { font-size: 2.6rem; margin-bottom: 0.85rem; opacity: 0.8; }
.dropzone-text {
    color: #64748b;
    font-size: 0.85rem;
    line-height: 1.75;
}
.dropzone-text strong { color: #2563eb; font-weight: 600; }

/* ── Image meta ── */
.img-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.9rem;
    background: #f0f7ff;
    border: 1px solid rgba(37,99,235,0.1);
    border-top: none;
    border-radius: 0 0 12px 12px;
    margin-top: -4px;
}
.img-meta-text {
    font-size: 0.68rem;
    color: #64748b;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace !important;
}
.img-status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #10b981;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.18);
}

/* ── Analyse button ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 0.91rem !important;
    padding: 0.85rem 1.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.25s !important;
    letter-spacing: 0.02em !important;
    margin-top: 0.85rem !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.35), 0 1px 0 rgba(255,255,255,0.15) inset !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.15) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 50%, #2563eb 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.45), 0 1px 0 rgba(255,255,255,0.15) inset !important;
}
.stButton > button:disabled {
    background: #dbe4f3 !important;
    color: #7488a8 !important;
    transform: none !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    border: 1px solid rgba(37,99,235,0.18) !important;
    text-shadow: none !important;
}

/* ── Result empty ── */
.result-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 340px;
    text-align: center;
    gap: 1rem;
    background: #ffffff;
    border-radius: 16px;
    border: 1.5px dashed rgba(37,99,235,0.22);
}
.result-empty-icon { font-size: 2.8rem; opacity: 0.25; }
.result-empty-text { font-size: 0.84rem; color: #94a3b8; line-height: 1.75; }

/* ── Prediction result ── */
.pred-header {
    background: #ffffff;
    border: 1px solid rgba(37,99,235,0.2);
    border-left: 4px solid #3b82f6;
    border-radius: 16px;
    padding: 1.4rem 1.55rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}
.pred-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 130px; height: 130px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(96,165,250,0.18), transparent 70%);
    pointer-events: none;
}
.pred-eyebrow {
    font-size: 0.6rem;
    font-weight: 700;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 0.45rem;
    font-family: 'JetBrains Mono', monospace !important;
}
.pred-class {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.9rem;
    font-weight: 800;
    color: #1e3a5f;
    letter-spacing: -0.03em;
    line-height: 1.15;
}
.pred-conf {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-top: 0.65rem;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    color: #059669;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 4px 13px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace !important;
}
.pred-conf-warn {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.25);
    color: #d97706;
}
.pred-conf-low {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.2);
    color: #dc2626;
}

/* ── Bar chart per kelas ── */
.bar-row { margin-bottom: 0.85rem; }
.bar-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.bar-cls { font-size: 0.8rem; color: #475569; font-weight: 600; }
.bar-pct {
    font-size: 0.75rem;
    color: #2563eb;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace !important;
}
.bar-track {
    height: 9px;
    background: #ffffff;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(37,99,235,0.18);
}
.bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #1d4ed8, #3b82f6, #60a5fa);
    background-size: 200% 100%;
    animation: barGrow 0.85s cubic-bezier(0.22,1,0.36,1) forwards,
               gradShift 3s ease infinite;
    animation-delay: var(--bar-delay, 0s), 0s;
    box-shadow: 0 0 10px rgba(37,99,235,0.28);
    position: relative;
    width: 0%;
}
.bar-fill::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 45%;
    background: rgba(255,255,255,0.3);
    border-radius: 8px 8px 0 0;
}
.bar-fill-dim {
    background: #bfdbfe;
    box-shadow: none;
    animation: barGrow 0.7s cubic-bezier(0.22,1,0.36,1) forwards;
    animation-delay: var(--bar-delay, 0s);
    width: 0%;
}
.bar-fill-dim::after { display: none; }


/* ── Divider ── */
.subtle-divider {
    border: none;
    border-top: 1px solid rgba(37,99,235,0.08);
    margin: 1.25rem 0;
}

/* ── Grad-CAM section (seksi terpisah, terpusat, di bawah kedua kolom) ── */
.st-key-gradcam_section {
    max-width: 900px;
    margin: 0 auto 3rem;
    padding: 2.25rem 2.5rem;
    background: #ffffff;
    border-radius: 20px;
    border: 1.5px solid rgba(37,99,235,0.18);
    box-shadow: 0 4px 22px rgba(37, 99, 235, 0.09),
                0 1px 4px rgba(37, 99, 235, 0.06);
    text-align: center;
}
.gradcam-section-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.25rem;
}
.gradcam-section-header .panel-icon-badge { margin: 0 auto; }
.gradcam-section-header .panel-title-group { align-items: center; text-align: center; }
.gradcam-info {
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.85;
    background: #f3f7fe;
    border: 1px solid rgba(37,99,235,0.14);
    border-radius: 14px;
    padding: 1rem 1.35rem;
    margin: 0 auto 1.5rem;
    max-width: 700px;
    text-align: left;
}
.gradcam-info strong { color: #1e3a5f; }
.gradcam-note {
    font-size: 0.78rem;
    color: #64748b;
    margin: 1.1rem auto 0;
    line-height: 1.75;
    background: #f8faff;
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: 12px;
    padding: 0.7rem 1.1rem;
    max-width: 700px;
    text-align: center;
}
.gradcam-note strong { color: #2563eb !important; }

/* ── Tentang Model & Threshold (expander) ── */
div[data-testid="stExpander"] {
    max-width: 1200px;
    margin: 0 auto 3rem;
    border: 1.5px solid rgba(37,99,235,0.18) !important;
    border-radius: 18px !important;
    background: #ffffff !important;
    box-shadow: 0 2px 16px rgba(37,99,235,0.06);
    overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #1e3a5f !important;
    font-size: 0.92rem !important;
    padding: 0.35rem 0.25rem !important;
}
div[data-testid="stExpander"] summary:hover { color: #2563eb !important; }

.about-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 0.75rem 0 1.5rem;
}
.about-item {
    background: #f3f7fe;
    border: 1px solid rgba(37,99,235,0.12);
    border-radius: 12px;
    padding: 0.85rem 1rem;
}
.about-item-label {
    font-size: 0.65rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace !important;
}
.about-item-value {
    font-size: 0.86rem;
    font-weight: 600;
    color: #1e3a5f;
}

.about-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.about-metric {
    text-align: center;
    background: linear-gradient(135deg, #eff6ff, #f0f9ff);
    border: 1px solid rgba(37,99,235,0.16);
    border-radius: 14px;
    padding: 1rem 0.5rem;
}
.about-metric-value {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: #2563eb;
}
.about-metric-label {
    font-size: 0.68rem;
    color: #64748b;
    font-weight: 600;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.about-note {
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.75;
    background: #f8faff;
    border: 1px solid rgba(37,99,235,0.14);
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1.5rem;
}
.about-note code {
    background: rgba(37,99,235,0.1);
    color: #2563eb;
    padding: 0.1rem 0.4rem;
    border-radius: 5px;
    font-size: 0.76rem;
}
.about-note strong { color: #1e3a5f; }

.about-subheader {
    font-size: 0.85rem;
    font-weight: 700;
    color: #1e3a5f;
    margin-bottom: 0.6rem;
}
.about-thresh-table {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.about-thresh-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #f8faff;
    border: 1px solid rgba(37,99,235,0.1);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.84rem;
    color: #1e3a5f;
    font-weight: 600;
}
.about-thresh-optimized {
    background: rgba(37,99,235,0.12);
    color: #2563eb;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700;
    padding: 0.2rem 0.65rem;
    border-radius: 8px;
    font-size: 0.8rem;
}
.about-thresh-default {
    background: rgba(148,163,184,0.15);
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600;
    padding: 0.2rem 0.65rem;
    border-radius: 8px;
    font-size: 0.8rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.75rem 2rem;
    border-top: 1px solid rgba(37,99,235,0.1);
    color: #94a3b8;
    font-size: 0.68rem;
    background: #ffffff;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Streamlit file uploader — distyle langsung pada elemen aslinya
       (tidak disembunyikan/ditumpuk lagi) supaya area yang terlihat
       dan area yang bisa diklik selalu 100% sama persis. ── */
div[data-testid="stFileUploader"] {
    background: transparent !important;
}
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploader"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.5rem !important;
    text-align: center !important;
    background: #ffffff !important;
    border: 1.5px dashed rgba(37,99,235,0.4) !important;
    border-radius: 16px !important;
    padding: 2rem 1.5rem !important;
    transition: all 0.25s !important;
}
div[data-testid="stFileUploader"] section:hover,
div[data-testid="stFileUploaderDropzone"]:hover,
div[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(37,99,235,0.6) !important;
    background: rgba(37,99,235,0.04) !important;
}
div[data-testid="stFileUploader"] svg {
    width: 26px !important;
    height: 26px !important;
    stroke: #3b82f6 !important;
    opacity: 0.85;
}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span {
    color: #64748b !important;
    font-size: 0.82rem !important;
}
/* Tombol Browse bawaan Streamlit — solid biru, jelas terlihat & tetap
   berada persis di tempat yang bisa diklik (bukan elemen palsu) */
div[data-testid="stFileUploader"] button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 0.55rem 1.25rem !important;
    margin-top: 0.3rem !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    text-shadow: none !important;
}
div[data-testid="stFileUploader"] button:hover {
    background: #1d4ed8 !important;
}
div[data-testid="stFileUploader"] button svg {
    stroke: #ffffff !important;
    opacity: 1;
    width: 15px !important;
    height: 15px !important;
}

.stSpinner > div { border-top-color: #3b82f6 !important; }
div[data-testid="stNotification"] { display: none; }
footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }

/* Rounded image corners */
div[data-testid="stImage"] img { border-radius: 12px 12px 0 0; }

/* ── Kotak preview gambar upload — tinggi selalu tetap, tidak ikut
       berubah walau ukuran/rasio gambar aslinya berbeda-beda, supaya
       tinggi panel kiri & kanan tetap seimbang. Grad-CAM (di panel
       kanan) tidak terkena aturan ini karena hanya menyasar kolom
       pertama. ── */
[data-testid="column"]:first-of-type [data-testid="stImage"] img,
[data-testid="stColumn"]:first-of-type [data-testid="stImage"] img {
    height: 360px !important;
    width: 100% !important;
    object-fit: cover !important;
    object-position: center !important;
}
</style>
""", unsafe_allow_html=True)

# ─── ASSETS ──────────────────────────────────────────────────────────────────
ASSETS_DIR       = "assets"
MODEL_PATH       = os.path.join(ASSETS_DIR, "model_mobilenet.h5")
CLASSES_PATH     = os.path.join(ASSETS_DIR, "classes.json")
THRESHOLDS_PATH  = os.path.join(ASSETS_DIR, "optimal_thresholds.json")
MODEL_INFO_PATH  = os.path.join(ASSETS_DIR, "model_info.json")
IMG_SIZE         = (224, 224)

CLASS_ICONS = {
    "Cyclone":    "🌀",
    "Earthquake": "🌍",
    "Flood":      "🌊",
    "Landslide":  "⛰️",
    "Wildfire":   "🔥",
}
DEFAULT_ICON = "⚠️"

# ─── PREPROCESSING ────────────────────────────────────────────────────────────
def preprocess_pil(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    resized = cv2.resize(arr, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0

# ─── LOAD ASSETS ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASSES_PATH) as f:
        return json.load(f)

@st.cache_data
def load_thresholds():
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH) as f:
            return json.load(f)
    return None

@st.cache_data
def load_model_info():
    """
    Info arsitektur/dataset/metrik model — opsional, dibaca dari
    assets/model_info.json (dibuat manual dari hasil notebook training,
    mis. cell G.1/G.3 di 'mobilenetv2_contributions'). Contoh isi file:
    {
        "fine_tune_layers": 30,
        "dataset_name": "Comprehensive Disaster Dataset (CDD)",
        "train_split": 70, "val_split": 20, "test_split": 10,
        "test_accuracy": 0.94, "macro_f1": 0.93, "mean_auc": 0.97,
        "class_counts": {"Fire_Disaster": 500, "...": 0}
    }
    Jika file belum ada, kembalikan None — UI akan pakai fallback aman.
    """
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH) as f:
            return json.load(f)
    return None

# ─── GRAD-CAM ─────────────────────────────────────────────────────────────────
def compute_gradcam(model, input_arr):
    try:
        # Cari backbone (MobileNetV2, >50 layer)
        backbone = None
        backbone_idx = None
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "layers") and len(layer.layers) > 50:
                backbone = layer
                backbone_idx = i
                break
        if backbone is None:
            return None

        # Model 1: outer_input → backbone.output (feature map)
        backbone_extractor = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=backbone.output,
        )

        # Model 2: backbone.output → prediksi akhir (head layers)
        bb_out_shape = backbone.output_shape[1:]
        head_inp = tf.keras.Input(shape=bb_out_shape)
        x = head_inp
        for layer in model.layers[backbone_idx + 1:]:
            x = layer(x)
        head_model = tf.keras.Model(inputs=head_inp, outputs=x)

        # Hitung Grad-CAM
        inp_tensor   = tf.cast(input_arr[np.newaxis, ...], tf.float32)
        # Langkah 1: dapatkan feature map DI LUAR tape
        backbone_out = backbone_extractor(inp_tensor, training=False)

        # Langkah 2: watch feature map SEBELUM forward pass head
        with tf.GradientTape() as tape:
            tape.watch(backbone_out)
            preds       = head_model(backbone_out, training=False)
            pred_idx    = tf.argmax(preds[0])
            class_score = preds[:, pred_idx]

        grads = tape.gradient(class_score, backbone_out)
        if grads is None:
            return None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = backbone_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        hm_r    = cv2.resize(heatmap.numpy(), IMG_SIZE)
        overlay = np.clip(cm.jet(hm_r)[:, :, :3] * 0.45 + input_arr * 0.55, 0, 1)
        return overlay

    except Exception:
        return None

# ─── NAVBAR ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <div class="dot"></div>
        🛰️&nbsp; DetekSi Bencana
    </div>
    <div style="display:flex;align-items:center;gap:0.85rem">
        <div class="nav-badge">MobileNetV2</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-ring-inner"></div>
    <div class="hero-eyebrow">🛰️ &nbsp; Disaster Detection</div>
    <h1>Klasifikasi Citra<br><span>Bencana Alam</span></h1>
    <p>Upload foto bencana, model akan mendeteksi jenis bencana secara otomatis
    menggunakan MobileNetV2 yang telah dilatih pada dataset CDD.</p>
</div>
""", unsafe_allow_html=True)

# ─── SPACER TIPIS ─────────────────────────────────────────────────────────────
st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

# ─── STATS BAR ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">5</div>
        <div class="stat-label">Kelas Bencana</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">MobileNetV2</div>
        <div class="stat-label">Arsitektur Model</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">224px</div>
        <div class="stat-label">Input Size</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">CDD</div>
        <div class="stat-label">Dataset</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── CECK ASSETS ─────────────────────────────────────────────────────────────
missing = [p for p in [MODEL_PATH, CLASSES_PATH] if not os.path.exists(p)]

# ─── MAIN AREA ───────────────────────────────────────────────────────────────
with st.container(key="main_area"):
    left_col, right_col = st.columns([1, 1], gap="large")

    # ════════════════════════════════════════
    # KOLOM KIRI — Upload
    # ════════════════════════════════════════
    with left_col:
        with st.container(border=True):
            st.markdown("""
            <div class="panel-header-main">
                <div class="panel-icon-badge">📤</div>
                <div class="panel-title-group">
                    <div class="panel-title">Upload Gambar</div>
                    <div class="panel-subtitle">Pilih citra bencana yang ingin dianalisis</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Pilih atau seret gambar ke sini",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            st.caption("Format: JPG, PNG &nbsp;·&nbsp; Maksimal 200 MB", unsafe_allow_html=True)

            if uploaded:
                pil_img = Image.open(uploaded)
                w, h    = pil_img.size
                size_kb = uploaded.size // 1024
                st.image(pil_img, use_container_width=True)
                st.markdown(f"""
                <div class="img-meta">
                    <span class="img-meta-text">
                        <span class="img-status-dot"></span>{uploaded.name}
                    </span>
                    <span class="img-meta-text">{w}×{h} &nbsp;·&nbsp; {size_kb} KB</span>
                </div>
                """, unsafe_allow_html=True)

            # Tombol analisis
            analyse_btn = st.button(
                "🔍  Analisis Gambar" if uploaded else "Pilih gambar terlebih dahulu",
                disabled=(uploaded is None or bool(missing)),
                use_container_width=True,
            )

    # ════════════════════════════════════════
    # KOLOM KANAN — Hasil
    # ════════════════════════════════════════
    with right_col:
        with st.container(border=True):
            st.markdown("""
            <div class="panel-header-main">
                <div class="panel-icon-badge">📊</div>
                <div class="panel-title-group">
                    <div class="panel-title">Hasil Klasifikasi</div>
                    <div class="panel-subtitle">Prediksi jenis bencana & tingkat keyakinan model</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            result_placeholder = st.empty()

            if not uploaded or not analyse_btn:
                result_placeholder.markdown("""
                <div class="result-empty">
                    <div class="result-empty-icon">🧠</div>
                    <div class="result-empty-text">
                        Hasil klasifikasi akan muncul di sini<br>
                        setelah gambar diunggah dan dianalisis.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif missing:
                missing_str = ", ".join(missing)
                result_placeholder.error(
                    f"File model tidak ditemukan: {missing_str}. "
                    "Letakkan file hasil training di folder assets/."
                )

            else:
                with st.spinner("Menganalisis gambar..."):
                    model      = load_model()
                    CLASS_NAMES = load_classes()
                    thresholds  = load_thresholds()

                    inp    = preprocess_pil(pil_img)
                    preds  = model.predict(inp[np.newaxis, ...], verbose=0)[0]

                    if thresholds:
                        adjusted = np.array(
                            [preds[i] / thresholds.get(c, 1.0) for i, c in enumerate(CLASS_NAMES)]
                        )
                        pred_idx = int(np.argmax(adjusted))
                    else:
                        pred_idx = int(np.argmax(preds))

                    pred_class = CLASS_NAMES[pred_idx]
                    pred_conf  = float(preds[pred_idx])
                    pred_icon  = CLASS_ICONS.get(pred_class, DEFAULT_ICON)

                # Badge warna berdasarkan confidence
                if pred_conf >= 0.75:
                    conf_class, conf_label = "pred-conf", "Tinggi"
                elif pred_conf >= 0.5:
                    conf_class, conf_label = "pred-conf pred-conf-warn", "Sedang"
                else:
                    conf_class, conf_label = "pred-conf pred-conf-low", "Rendah"

                # Header prediksi
                result_placeholder.markdown(f"""
                <div class="pred-header">
                    <div class="pred-eyebrow">Jenis Bencana Terdeteksi</div>
                    <div class="pred-class">{pred_icon}&nbsp; {pred_class}</div>
                    <span class="{conf_class}">Confidence {pred_conf*100:.1f}% — {conf_label}</span>
                </div>
                <hr class="subtle-divider"/>
                <div class="panel-header" style="margin-bottom:0.75rem;padding-bottom:0.65rem">
                    <div class="panel-header-dot"></div>
                    <div class="panel-label">Distribusi Probabilitas</div>
                </div>
                """, unsafe_allow_html=True)

                # Bar chart probabilitas per kelas (HTML custom)
                sorted_preds = sorted(zip(CLASS_NAMES, preds), key=lambda x: x[1], reverse=True)
                bars_html = ""
                for idx, (cls, prob) in enumerate(sorted_preds):
                    pct = prob * 100
                    is_top = (cls == pred_class)
                    fill_class = "bar-fill" if is_top else "bar-fill-dim"
                    cls_icon = CLASS_ICONS.get(cls, "•")
                    delay = idx * 0.1
                    bars_html += f"""
                    <div class="bar-row">
                        <div class="bar-label-row">
                            <span class="bar-cls">{cls_icon} {cls}</span>
                            <span class="bar-pct">{pct:.1f}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="{fill_class}"
                                 style="--bar-w:{pct:.1f}%;--bar-delay:{delay:.2f}s"></div>
                        </div>
                    </div>"""

                st.markdown(bars_html, unsafe_allow_html=True)

# ─── GRAD-CAM — seksi terpisah, di tengah, di bawah kedua kolom ─────────────
if uploaded and analyse_btn and not missing:
    with st.container(key="gradcam_section"):
        st.markdown("""
        <div class="gradcam-section-header">
            <div class="panel-icon-badge">🔥</div>
            <div class="panel-title-group">
                <div class="panel-title">Grad-CAM — Area Fokus Model</div>
                <div class="panel-subtitle">Visualisasi area gambar yang paling memengaruhi keputusan model</div>
            </div>
        </div>
        <div class="gradcam-info">
            <strong>Apa itu Grad-CAM?</strong> Grad-CAM (Gradient-weighted Class Activation
            Mapping) adalah teknik visualisasi yang menunjukkan bagian mana dari gambar yang
            paling berpengaruh saat model AI mengambil keputusan klasifikasi. Warna
            <strong style="color:#dc2626">merah/kuning</strong> menandakan area dengan pengaruh
            tinggi terhadap keputusan model, sedangkan warna
            <strong style="color:#2563eb">biru/ungu</strong> menandakan area dengan pengaruh rendah.
        </div>
        """, unsafe_allow_html=True)

        try:
            overlay = compute_gradcam(model, inp)
            if overlay is not None:
                gc_left, gc_mid, gc_right = st.columns([1, 2.2, 1])
                with gc_mid:
                    st.image(overlay, use_container_width=True, clamp=True)
                st.markdown(
                    f'<div class="gradcam-note">Area warna merah/kuning adalah region '
                    f'yang paling berpengaruh dalam keputusan model mengklasifikasi gambar '
                    f'sebagai <strong>{pred_class}</strong>.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Grad-CAM tidak tersedia untuk arsitektur model ini.")
        except Exception as e:
            st.markdown(
                f'<div class="gradcam-note">Grad-CAM tidak tersedia: {e}</div>',
                unsafe_allow_html=True
            )

# ─── TENTANG MODEL & THRESHOLD ───────────────────────────────────────────────
if not missing:
    model_info = load_model_info()
    thresholds_info = load_thresholds()
    CLASS_NAMES_INFO = load_classes()

    with st.expander("ℹ️  Tentang Model & Threshold Klasifikasi", expanded=False):
        # ── Info arsitektur dasar (selalu tersedia) ──
        finetune_txt = (
            f"{model_info['fine_tune_layers']} layer terakhir"
            if model_info and "fine_tune_layers" in model_info
            else "—"
        )
        dataset_txt = (
            model_info.get("dataset_name", "Comprehensive Disaster Dataset (CDD)")
            if model_info else "Comprehensive Disaster Dataset (CDD)"
        )
        split_txt = (
            f"{model_info.get('train_split',70)}% / {model_info.get('val_split',20)}% / {model_info.get('test_split',10)}%"
            if model_info else "70% / 20% / 10%"
        )

        st.markdown(f"""
        <div class="about-grid">
            <div class="about-item">
                <div class="about-item-label">Arsitektur</div>
                <div class="about-item-value">MobileNetV2 (Transfer Learning)</div>
            </div>
            <div class="about-item">
                <div class="about-item-label">Ukuran Input</div>
                <div class="about-item-value">{IMG_SIZE[0]}×{IMG_SIZE[1]} px</div>
            </div>
            <div class="about-item">
                <div class="about-item-label">Fine-tuning</div>
                <div class="about-item-value">{finetune_txt}</div>
            </div>
            <div class="about-item">
                <div class="about-item-label">Dataset</div>
                <div class="about-item-value">{dataset_txt}</div>
            </div>
            <div class="about-item">
                <div class="about-item-label">Split Train/Val/Test</div>
                <div class="about-item-value">{split_txt}</div>
            </div>
            <div class="about-item">
                <div class="about-item-label">Jumlah Kelas</div>
                <div class="about-item-value">{len(CLASS_NAMES_INFO)} kelas bencana</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrik evaluasi (hanya tampil jika model_info.json tersedia) ──
        if model_info and all(k in model_info for k in ("test_accuracy", "macro_f1", "mean_auc")):
            st.markdown(f"""
            <div class="about-metrics">
                <div class="about-metric">
                    <div class="about-metric-value">{model_info['test_accuracy']*100:.1f}%</div>
                    <div class="about-metric-label">Test Accuracy</div>
                </div>
                <div class="about-metric">
                    <div class="about-metric-value">{model_info['macro_f1']:.3f}</div>
                    <div class="about-metric-label">Macro F1-Score</div>
                </div>
                <div class="about-metric">
                    <div class="about-metric-value">{model_info['mean_auc']:.3f}</div>
                    <div class="about-metric-label">Mean AUC</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="about-note">
                Metrik evaluasi (accuracy, F1, AUC) belum tersedia. Tambahkan file
                <code>assets/model_info.json</code> berisi hasil evaluasi test set
                dari notebook training untuk menampilkannya di sini.
            </div>
            """, unsafe_allow_html=True)

        # ── Info threshold per kelas ──
        st.markdown("""
        <div class="about-subheader">🎯 &nbsp;Threshold Klasifikasi per Kelas</div>
        <div class="about-note" style="margin-bottom:1rem">
            Untuk mengatasi ketidakseimbangan jumlah data antar kelas, setiap kelas
            memakai <strong>threshold optimal</strong> hasil optimasi kurva ROC
            (Youden's J statistic) — bukan threshold default 0.5 — supaya kelas
            minoritas tidak sulit terdeteksi.
        </div>
        """, unsafe_allow_html=True)

        rows_html = ""
        for cls in CLASS_NAMES_INFO:
            icon = CLASS_ICONS.get(cls, DEFAULT_ICON)
            if thresholds_info and cls in thresholds_info:
                t_val   = thresholds_info[cls]
                t_badge = f'<span class="about-thresh-optimized">{t_val:.3f}</span>'
            else:
                t_badge = '<span class="about-thresh-default">0.500 (default)</span>'
            rows_html += f"""
            <div class="about-thresh-row">
                <span>{icon} {cls}</span>
                {t_badge}
            </div>"""

        st.markdown(f'<div class="about-thresh-table">{rows_html}</div>', unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🛰️ &nbsp; DetekSi Bencana &nbsp;·&nbsp; MobileNetV2 + CDD Dataset &nbsp;·&nbsp;
    Tugas Akhir &mdash; Universitas Dian Nuswantoro
</div>
""", unsafe_allow_html=True)