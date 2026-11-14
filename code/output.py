import os
import sys
import json
import math
import time
import datetime
import subprocess
import cv2
import numpy as np
from fpdf import FPDF

# ── Design tokens (BGR) ───────────────────────────────────────────────────────
# Calm, professional dark theme (the look of Linear / Stripe / VS Code):
# deep blue-grey surfaces, ONE accent colour (blue), and green / amber / red used
# only for meaning (good / check / problem). Hex values are RGB.
DARK    = (23,  17,  13)   # #0D1117 app background
PANEL   = (34,  26,  19)   # #131A22 raised surface (side panels, bars)
CARD    = (46,  35,  26)   # #1A232E cards
CARD_HI = (64,  48,  34)   # #223040 hovered / selected card
BORDER  = (68,  54,  42)   # #2A3644 hairline borders
GOLD    = (246, 141,  76)  # #4C8DF6 PRIMARY accent (blue; old name kept for the code)
GOLD_HI = (255, 174, 122)  # #7AAEFF accent highlight
TEAL    = (207, 184,  34)  # #22B8CF secondary accent (info)
GREEN   = (123, 199,  52)  # #34C77B success / good
AMBER   = (60,  177, 242)  # #F2B13C check / warning
RED     = (91,   82, 240)  # #F0525B problem / danger
WHITE   = (243, 237, 232)  # #E8EDF3 primary text
GREY    = (176, 161, 147)  # #93A1B0 secondary text
DIM     = (122, 107,  92)  # #5C6B7A tertiary text / placeholders

FONT     = cv2.FONT_HERSHEY_SIMPLEX
FONT_DUP = cv2.FONT_HERSHEY_DUPLEX

# Type scale (≈ px = 30 × value): caption 12 · label 13 · body 14 · title 18 · heading 26
TS_CAP, TS_LABEL, TS_BODY, TS_TITLE, TS_H1 = 0.40, 0.44, 0.48, 0.60, 0.86

# Sharp Segoe UI text instead of OpenCV's stroke fonts (ui_text.py)
from ui_text import put_text as _put_text, text_size as _text_size
import ui_text as _ui

# Clickable areas registered while a screen is drawn: [((x1, y1, x2, y2), key)].
# The app turns a click inside one into that key press (main._wait_key), so every
# button that shows a shortcut also works with the mouse.
CLICK_TARGETS = []
_KEYCODES = {'Enter': 13, 'Esc': 27, 'Tab': 9,
             # pseudo-keys: clicking an input box focuses it (handled by main.py)
             'focus:name': 200, 'focus:height': 201, 'focus:weight': 202, 'focus:age': 203}


def _register(rect, key):
    code = _KEYCODES.get(key, ord(key.lower()) if isinstance(key, str) and len(key) == 1 else None)
    if code is not None:
        CLICK_TARGETS.append((tuple(int(v) for v in rect), code))

# Per-measurement accent colors (BGR) — cyan accent, green for circumferences
_ACCENTS = {k: GOLD for k in ('shoulder_width', 'chest', 'waist', 'back_length',
                               'shirt_length', 'sleeve_length', 'neck')}   # one calm accent

MEASUREMENT_ORDER = [
    ('shoulder_width', 'Shoulder Width'),
    ('chest',          'Chest Circ.'),
    ('waist',          'Waist Circ.'),
    ('back_length',    'Back Length'),
    ('shirt_length',   'Shirt Length'),
    ('sleeve_length',  'Sleeve Length'),
    ('neck',           'Neck Circ.'),
]

# ── Multi-country garment size system ────────────────────────────────────────
#
# Body measurement ≠ Garment measurement.
# Industry standard EASE values (added to body to get garment size):
#   T-shirt chest: +4–6 cm   Dress shirt chest: +8–10 cm
#   Casual shirt:  +6–8 cm   Jacket/blazer:    +10–14 cm
#   Trousers waist: +2–4 cm
#
# All size charts below use BODY measurements (cm) as input.
# Source references:
#   ISO 3636 / EN 13402 (EU/UK), ASTM D5585 (US), JIS L 4004 (JP),
#   GB/T 1335.1 (CN), local BD market surveys (chest inches standard).

# ISO/EU letter sizes from body measurements (men, unisex fallback)
_SIZE_CHART = {
    'chest':          [('XS',(76,84)),('S',(84,92)),('M',(92,100)),('L',(100,108)),
                       ('XL',(108,116)),('2XL',(116,124)),('3XL',(124,999))],
    'waist':          [('XS',(58,66)),('S',(66,74)),('M',(74,82)),('L',(82,90)),
                       ('XL',(90,98)),('2XL',(98,106)),('3XL',(106,999))],
    'hip':            [('XS',(80,88)),('S',(88,96)),('M',(96,104)),('L',(104,112)),
                       ('XL',(112,120)),('2XL',(120,128)),('3XL',(128,999))],
    'shoulder_width': [('XS',(33,36)),('S',(36,39)),('M',(39,42)),('L',(42,45)),
                       ('XL',(45,49)),('2XL',(49,53)),('3XL',(53,999))],
    'neck':           [('XS',(30,33)),('S',(33,36)),('M',(36,39)),('L',(39,42)),
                       ('XL',(42,46)),('2XL',(46,50)),('3XL',(50,999))],
    'sleeve_length':  [('XS',(56,59)),('S',(59,62)),('M',(62,65)),('L',(65,68)),
                       ('XL',(68,71)),('2XL',(71,74)),('3XL',(74,999))],
    'back_length':    [('XS',(40,43)),('S',(43,46)),('M',(46,49)),('L',(49,52)),
                       ('XL',(52,55)),('2XL',(55,58)),('3XL',(58,999))],
    'shirt_length':   [('XS',(63,66)),('S',(66,70)),('M',(70,74)),('L',(74,78)),
                       ('XL',(78,82)),('2XL',(82,86)),('3XL',(86,999))],
}

# ── Per-country size tables (all keyed by body chest cm as primary) ───────────
# Each table: list of (display_label, chest_lo, chest_hi) from body measurement
# Secondary rows use hip/waist where relevant.

_COUNTRY_SIZES = {
    # Bangladesh local ready-made — sold by garment chest in inches (body+~5cm ease)
    'BD': {
        'label': 'Bangladesh',
        'chest': [
            ('34"', 78, 83), ('36"', 83, 88), ('38"', 88, 93),
            ('40"', 93, 98), ('42"', 98, 103), ('44"', 103, 109),
            ('46"', 109, 115), ('48"', 115, 121), ('50"', 121, 999),
        ],
        'waist': [
            ('28"', 65, 70), ('30"', 70, 75), ('32"', 75, 80),
            ('34"', 80, 85), ('36"', 85, 91), ('38"', 91, 97),
            ('40"', 97, 103), ('42"', 103, 999),
        ],
        'neck': [
            ('13"', 30, 33), ('13.5"', 33, 35), ('14"', 35, 36),
            ('14.5"', 36, 37), ('15"', 37, 38), ('15.5"', 38, 40),
            ('16"', 40, 42), ('16.5"', 42, 44), ('17"', 44, 999),
        ],
        'ease_note': 'Garment chest = body + ~5 cm ease',
    },

    # United States — ASTM D5585 & common retail (body measurements, men's)
    'US': {
        'label': 'United States',
        'chest': [
            ('XS', 76, 84), ('S', 84, 94), ('M', 94, 104), ('L', 104, 114),
            ('XL', 114, 124), ('2XL', 124, 134), ('3XL', 134, 999),
        ],
        'waist': [
            ('28', 70, 74), ('30', 74, 78), ('32', 78, 83), ('34', 83, 88),
            ('36', 88, 93), ('38', 93, 99), ('40', 99, 105), ('42', 105, 999),
        ],
        'neck': [
            ('14"', 34, 36), ('14.5"', 36, 37), ('15"', 37, 38),
            ('15.5"', 38, 39), ('16"', 39, 41), ('16.5"', 41, 42),
            ('17"', 42, 44), ('17.5"', 44, 999),
        ],
        'ease_note': 'US retail adds 4-10 cm to body for garment fit',
    },

    # United Kingdom — BS 6184 / EN 13402 retail (similar to US but slightly smaller runs)
    'UK': {
        'label': 'United Kingdom',
        'chest': [
            ('XS', 76, 84), ('S', 84, 94), ('M', 94, 104), ('L', 104, 114),
            ('XL', 114, 124), ('2XL', 124, 134), ('3XL', 134, 999),
        ],
        'waist': [
            ('28', 70, 74), ('30', 74, 78), ('32', 78, 83), ('34', 83, 88),
            ('36', 88, 93), ('38', 93, 99), ('40', 99, 105), ('42', 105, 999),
        ],
        'neck': [
            ('14"', 34, 36), ('14.5"', 36, 37), ('15"', 37, 38),
            ('15.5"', 38, 39), ('16"', 39, 41), ('16.5"', 41, 42),
            ('17"', 42, 44), ('17.5"', 44, 999),
        ],
        'ease_note': 'UK sizes identical to US letter sizes for men',
    },

    # Europe — EN 13402 numeric (body chest cm ranges), continental system
    'EU': {
        'label': 'Europe (EN 13402)',
        'chest': [
            ('44', 84, 88), ('46', 88, 92), ('48', 92, 96), ('50', 96, 100),
            ('52', 100, 104), ('54', 104, 108), ('56', 108, 112),
            ('58', 112, 116), ('60', 116, 120), ('62', 120, 999),
        ],
        'waist': [
            ('44', 76, 80), ('46', 80, 84), ('48', 84, 88), ('50', 88, 92),
            ('52', 92, 96), ('54', 96, 100), ('56', 100, 104),
            ('58', 104, 108), ('60', 108, 999),
        ],
        'neck': [
            ('36', 35, 36), ('37', 36, 37), ('38', 37, 38), ('39', 38, 39),
            ('40', 39, 40), ('41', 40, 41), ('42', 41, 42), ('43', 42, 43),
            ('44', 43, 999),
        ],
        'ease_note': 'EU numeric = body chest cm (±2 cm tolerance)',
    },

    # Japan — JIS L 4004 (2019) men's tops, body measurements
    'JP': {
        'label': 'Japan (JIS L4004)',
        'chest': [
            ('SS', 76, 84), ('S', 84, 92), ('M', 92, 100), ('L', 100, 108),
            ('XL', 108, 116), ('2L', 116, 124), ('3L', 124, 132), ('4L', 132, 999),
        ],
        'waist': [
            ('SS', 60, 67), ('S', 67, 74), ('M', 74, 82), ('L', 82, 90),
            ('XL', 90, 98), ('2L', 98, 106), ('3L', 106, 999),
        ],
        'neck': [
            ('S', 33, 36), ('M', 36, 39), ('L', 39, 42), ('XL', 42, 999),
        ],
        'ease_note': 'JIS uses body measurements; garment adds 6-12 cm ease',
    },

    # China — GB/T 1335.1-2008 men's (A-type = average proportions)
    'CN': {
        'label': 'China (GB/T 1335)',
        'chest': [
            ('84A', 82, 86), ('88A', 86, 90), ('92A', 90, 94), ('96A', 94, 98),
            ('100A', 98, 102), ('104A', 102, 106), ('108A', 106, 110),
            ('112A', 110, 114), ('116A', 114, 118), ('120A', 118, 999),
        ],
        'waist': [
            ('68A', 66, 70), ('72A', 70, 74), ('76A', 74, 78), ('80A', 78, 82),
            ('84A', 82, 86), ('88A', 86, 90), ('92A', 90, 94), ('96A', 94, 999),
        ],
        'neck': [
            ('36', 35, 37), ('37', 36, 38), ('38', 37, 39), ('39', 38, 40),
            ('40', 39, 41), ('41', 40, 42), ('42', 41, 43), ('43', 42, 999),
        ],
        'ease_note': 'GB/T label = body chest cm; A = average body proportion',
    },

    # India — IS 1197 / BIS (similar to EU letter sizes but with local chart)
    'IN': {
        'label': 'India (BIS IS 1197)',
        'chest': [
            ('38', 82, 87), ('40', 87, 92), ('42', 92, 97), ('44', 97, 102),
            ('46', 102, 107), ('48', 107, 112), ('50', 112, 118), ('52', 118, 999),
        ],
        'waist': [
            ('28"', 65, 70), ('30"', 70, 75), ('32"', 75, 80), ('34"', 80, 85),
            ('36"', 85, 91), ('38"', 91, 97), ('40"', 97, 999),
        ],
        'neck': [
            ('14"', 34, 36), ('14.5"', 36, 37), ('15"', 37, 38),
            ('15.5"', 38, 39), ('16"', 39, 41), ('16.5"', 41, 999),
        ],
        'ease_note': 'India uses garment chest inches (body + ~2 cm ease)',
    },
}

COUNTRY_KEYS = list(_COUNTRY_SIZES.keys())  # order for cycling

# Standard garment ease (cm) per field and garment type, for educational display
_EASE_TABLE = {
    't_shirt':  {'chest': 5,  'waist': 3,  'hip': 4},
    'shirt':    {'chest': 8,  'waist': 4,  'hip': 4},
    'sweater':  {'chest': 10, 'waist': 6,  'hip': 6},
    'jacket':   {'chest': 12, 'waist': 8,  'hip': 8},
}

def _country_size(country, field, value_cm):
    """Look up size label for a given country, field and body measurement (cm)."""
    ctable = _COUNTRY_SIZES.get(country, {})
    for label, lo, hi in ctable.get(field, []):
        if lo <= value_cm < hi:
            return label
    return ''

def _size_label(key, value_cm):
    """ISO/EU letter size label from body measurement — used for quick badge display."""
    return _iso_letter(key, value_cm)

# Standard ease per field (cm) for estimating garment size
_STANDARD_EASE = {
    'chest':         6.0,   # casual shirt
    'waist':         4.0,   # trouser
    'hip':           4.0,
    'neck':          2.0,
    'shoulder_width':0.0,
    'back_length':   0.5,
    'sleeve_length': 1.5,
    'shirt_length':  2.0,
}

def _garment_cm(key, value_cm):
    """Body measurement + standard ease = what the GARMENT actually measures."""
    if value_cm is None:
        return None
    return round(value_cm + _STANDARD_EASE.get(key, 0.0), 1)

def _inch_size(key, value_cm):
    """
    BD/South Asian ready-made size.
    Circumference: body + ease → nearest inch (the label on the tag).
    Length: body in inches.
    """
    if value_cm is None:
        return ''
    g_cm = _garment_cm(key, value_cm)
    if key in ('chest', 'waist', 'hip', 'neck'):
        return f'{int(round(g_cm / 2.54))}'   # plain number, no " mark
    elif key in ('sleeve_length', 'back_length', 'shirt_length'):
        return f'{g_cm / 2.54:.1f}"'
    return ''

def _iso_letter(key, value_cm):
    """ISO/EU letter size (XS/S/M/L/XL/2XL/3XL) from body measurement."""
    if value_cm is None:
        return ''
    return next((n for n, (lo, hi) in _SIZE_CHART.get(key, []) if lo <= value_cm < hi), '')

def _garment_recommendation(measurements, country='BD'):
    """
    Returns a short garment-size summary string for given country.
    """
    chest = measurements.get('chest')
    waist = measurements.get('waist')
    parts = []
    if chest:
        sz = _country_size(country, 'chest', chest)
        if sz:
            parts.append(f'Top: {sz}')
    if waist:
        sz = _country_size(country, 'waist', waist)
        if sz:
            parts.append(f'Bottom: {sz}')
    return '   |   '.join(parts)


# ── Drawing helpers ────────────────────────────────────────────────────────────
def _rrect(frame, x1, y1, x2, y2, r, color):
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    r = max(0, min(int(r), (x2 - x1) // 2, (y2 - y1) // 2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)


def _rrect_border(frame, x1, y1, x2, y2, r, color, thick=1):
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    r = max(0, min(int(r), (x2 - x1) // 2, (y2 - y1) // 2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.line(frame, (x1+r,y1), (x2-r,y1), color, thick, cv2.LINE_AA)
    cv2.line(frame, (x1+r,y2), (x2-r,y2), color, thick, cv2.LINE_AA)
    cv2.line(frame, (x1,y1+r), (x1,y2-r), color, thick, cv2.LINE_AA)
    cv2.line(frame, (x2,y1+r), (x2,y2-r), color, thick, cv2.LINE_AA)
    for (cx,cy,s) in [(x1+r,y1+r,180),(x2-r,y1+r,270),(x1+r,y2-r,90),(x2-r,y2-r,0)]:
        cv2.ellipse(frame,(cx,cy),(r,r),0,s,s+90,color,thick,cv2.LINE_AA)


def _alpha(frame, x1, y1, x2, y2, r, color, a):
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    if x2 <= x1 or y2 <= y1:
        return
    ov = frame.copy()
    _rrect(ov, x1, y1, x2, y2, r, color)
    cv2.addWeighted(ov, a, frame, 1 - a, 0, frame)


def _text(frame, msg, pos, color=WHITE, scale=0.65, thick=2, font=FONT):
    # Ensure integer pixel coordinates — float coords silently crash in newer OpenCV
    _put_text(frame, str(msg), (int(pos[0]), int(pos[1])), font, scale, color, thick)


def _centered(frame, msg, cy, color=WHITE, scale=0.72, thick=2, font=FONT):
    w = frame.shape[1]
    tw = _text_size(msg, font, scale, thick)[0][0]
    _text(frame, msg, ((w-tw)//2, cy), color, scale, thick, font)


# ── Premium UI animation helpers ───────────────────────────────────────────────
def _ease_out(t):
    t = max(0.0, min(1.0, float(t)))
    return 1.0 - (1.0 - t) ** 3

def _phase(elapsed, t0, t1):
    """Ease-out progress 0→1 within the time window [t0, t1]."""
    if elapsed <= t0: return 0.0
    if elapsed >= t1: return 1.0
    return _ease_out((elapsed - t0) / (t1 - t0))

def _dot_bg(frame, spacing=30):
    """
    Replaced with a cleaner subtle gradient vignette.
    The old dot-matrix pattern looked amateurish; this is invisible unless you look.
    Kept the same name so all callers still work.
    """
    # Very subtle top-to-bottom darkening — professional vignette, no dots
    fh, fw = frame.shape[:2]
    for row in range(0, fh, 6):          # stride 6 — fast enough, invisible seams
        alpha = 0.04 * (row / fh)        # darkens slightly toward bottom (0→4%)
        frame[row:row+6] = (frame[row:row+6] * (1.0 - alpha)).astype(frame.dtype)

def _glow(frame, cx, cy, r, color, alpha):
    if r < 1: return
    ov = frame.copy()
    cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)

def _diamond(frame, cx, cy, sz, color):
    if sz < 1: return
    pts = np.array([[cx, cy-sz],[cx+sz*6//10, cy],
                    [cx, cy+sz],[cx-sz*6//10, cy]], np.int32)
    cv2.fillPoly(frame, [pts], color)

def _avatar_circle(frame, cx, cy, r, name):
    """Colored circle with first initial of name."""
    if not name: name = '?'
    hv = sum(ord(c) * (i+1) for i, c in enumerate(name[:6]))
    PALETTE = [(160,80,50),(55,140,60),(160,60,200),(30,120,200),(160,180,40),(120,50,170)]
    col = PALETTE[hv % len(PALETTE)]
    cv2.circle(frame, (cx, cy), r, col, -1)
    cv2.circle(frame, (cx, cy), r, (200, 200, 220), 1)
    ini = name[0].upper()
    tw, th = _text_size(ini, FONT_DUP, 0.44, 1)[0]
    _put_text(frame, ini, (cx - tw//2, cy + th//2 + 1), FONT_DUP, 0.44, (240, 240, 242), 1)


# ── Glassmorphism card helper ──────────────────────────────────────────────────
def _glassmorphism_card(frame, x1, y1, x2, y2, r=10, blur_ksize=15, alpha=0.72):
    """
    Frosted-glass card: blurs the region behind the card, applies a dark
    semi-transparent panel, then draws a subtle glowing border.
    Gives every card a premium 2025 Liquid Glass appearance.
    """
    fh, fw = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(fw - 1, x2), min(fh - 1, y2)
    if x2c <= x1c or y2c <= y1c:
        return
    roi = frame[y1c:y2c, x1c:x2c].copy()
    k   = max(3, blur_ksize | 1)          # must be odd
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    dark_ov = np.full_like(blurred, CARD, dtype=np.uint8)
    mixed   = cv2.addWeighted(blurred, 1.0 - alpha, dark_ov, alpha, 0)
    frame[y1c:y2c, x1c:x2c] = mixed
    _rrect_border(frame, x1, y1, x2, y2, r, BORDER, 1)


# ── Loyalty tier helpers ───────────────────────────────────────────────────────
# Points: New<100, Regular 100-499, Gold 500-999, VIP 1000+
_LOYALTY_TIERS = [
    (1000, 'VIP',     (0, 210, 255), '★'),   # bright amber-gold
    ( 500, 'GOLD',    (0, 165, 240), '◆'),   # amber
    ( 100, 'REGULAR', TEAL,          '●'),
    (   0, 'NEW',     GREY,          '○'),
]

def _loyalty_tier(points):
    """Return (tier_name, bgr_color, icon_char) for a points value."""
    for min_pts, name, col, icon in _LOYALTY_TIERS:
        if points >= min_pts:
            return name, col, icon
    return 'NEW', GREY, '○'


def draw_loyalty_badge(frame, x, y, points):
    """Draw a compact loyalty tier chip at (x, y). Width auto-fits the tier name."""
    tier_name, tier_col, icon = _loyalty_tier(points)
    tw  = _text_size(tier_name, FONT, 0.30, 1)[0][0]
    bw  = tw + 26
    bh  = 18
    _rrect(frame, x, y, x + bw, y + bh, 5, DARK)
    _rrect_border(frame, x, y, x + bw, y + bh, 5, tier_col, 1)
    _put_text(frame, icon, (x + 4, y + 13), FONT, 0.28, tier_col, 1)
    _text(frame, tier_name, (x + 14, y + 13), tier_col, 0.30, 1)
    return bw


# ── Style recommendation engine ────────────────────────────────────────────────
_STYLE_TIPS = {
    'hourglass': [
        'Belt-waisted dresses accentuate balanced proportions',
        'Wrap shirts & fitted blazers complement your shape',
    ],
    'pear': [
        'Wide-collar or V-neck tops add width to shoulders',
        'A-line skirts & pleated trousers balance the hips',
    ],
    'apple': [
        'Empire-waist tops draw attention upward',
        'V-necks & vertical stripes elongate the torso',
    ],
    'inverted': [
        'Wide-leg trousers balance broad shoulders',
        'A-line / pleated bottoms add volume at hip',
    ],
    'rectangle': [
        'Belted pieces create a defined waist illusion',
        'Peplum tops and ruffled hips add curves',
    ],
    'unknown': [
        'Complete 3-view scan for body type detection',
    ],
}

def get_style_recommendation(body_type, garment_type=None):
    """Return 1-2 style tip strings for the detected body type."""
    return _STYLE_TIPS.get(body_type, _STYLE_TIPS['unknown'])


def draw_style_tip_overlay(frame, body_type, x=None, y=None):
    """
    Draw a small style recommendation chip at the bottom of the results area.
    Returns the y coordinate after the chip.
    """
    tips = get_style_recommendation(body_type)
    h, w = frame.shape[:2]
    ox = x if x is not None else 14
    oy = y if y is not None else h - 90
    _rrect(frame, ox, oy, ox + w - ox - 14, oy + 22, 6, (8, 18, 30))
    _rrect_border(frame, ox, oy, ox + w - ox - 14, oy + 22, 6, TEAL, 1)
    _text(frame, 'STYLE TIP:', (ox + 8, oy + 15), TEAL, 0.34, 1, FONT_DUP)
    tip_txt = tips[0] if tips else ''
    _text(frame, tip_txt[:72], (ox + 82, oy + 15), WHITE, 0.34, 1)
    return oy + 28


# ── Splash / intro screen ──────────────────────────────────────────────────────
def draw_splash_screen(frame, elapsed):
    """
    Short startup screen: logo, name, loading bar. Returns (frame, done);
    done after 3 s (any key skips, handled by the caller).
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2 - 30
    frame[:] = DARK
    t = _ease_out(min(1.0, elapsed / 0.8))
    s = int(36 * (0.85 + 0.15 * t))
    if t > 0.05:
        _rrect(frame, cx - s, cy - 70 - s, cx + s, cy - 70 + s, int(s * 0.32), GOLD)
        tw = _text_size('T', FONT, 1.6 * s / 36, 3)[0]
        _text(frame, 'T', (cx - tw[0] // 2, cy - 70 + tw[1] // 2), (255, 255, 255), 1.6 * s / 36, 3)
    a = _ease_out(min(1.0, max(0.0, (elapsed - 0.3) / 0.6)))
    if a > 0:
        col = tuple(int(DARK[i] + (WHITE[i] - DARK[i]) * a) for i in range(3))
        sub = tuple(int(DARK[i] + (GREY[i] - DARK[i]) * a) for i in range(3))
        _centered(frame, 'TailorCam', cy + 26, col, 1.25, 3)
        _centered(frame, 'Body measurement studio', cy + 60, sub, TS_BODY + 0.04, 1)
    bw, by = 220, cy + 110
    p = min(1.0, elapsed / 2.8)
    _rrect(frame, cx - bw // 2, by, cx + bw // 2, by + 4, 2, CARD_HI)
    _rrect(frame, cx - bw // 2, by, cx - bw // 2 + max(4, int(bw * p)), by + 4, 2, GOLD)
    msg = 'Loading body models…' if p < 1 else 'Ready'
    _centered(frame, msg, by + 30, DIM, TS_CAP, 1)
    return frame, elapsed >= 3.0


# ── Body silhouette for results left panel ────────────────────────────────────
def _body_silhouette(frame, cx, cy, scale, measurements=None):
    """Body diagram for results panel. measurements=dict of key→cm values."""
    def pt(x, y):
        return (int(cx + x * scale / 100), int(cy + y * scale / 100))

    BDY = (75, 55, 30)    # warm-toned skeleton lines
    ANN = (0, 165, 240)   # amber annotation lines (GOLD)
    LBL = (0, 120, 180)   # dim amber label text

    def _val(key):
        if measurements:
            v = measurements.get(key)
            if v is not None:
                return f'{v:.0f}'
        return ''

    # skeleton with glow
    for thickness, alpha_v in [(6, 0.06), (4, 0.12)]:
        ov = frame.copy()
        cv2.ellipse(ov, pt(0,-82), (int(16*scale/100), int(20*scale/100)), 0,0,360,BDY,thickness)
        lines = [
            (pt(-5,-62), pt(-5,-50)), (pt(5,-62), pt(5,-50)),
            (pt(-5,-52), pt(-42,-46)), (pt(5,-52), pt(42,-46)),
            (pt(-42,-46), pt(-44,8)), (pt(42,-46), pt(44,8)),
            (pt(-44,8), pt(-40,50)), (pt(44,8), pt(40,50)),
            (pt(-42,-46), pt(-28,18)), (pt(42,-46), pt(28,18)),
            (pt(-28,18), pt(-35,52)), (pt(28,18), pt(35,52)),
            (pt(-35,52), pt(35,52)),
            (pt(-23,52), pt(-21,100)), (pt(23,52), pt(21,100)),
            (pt(-35,52), pt(-23,52)), (pt(35,52), pt(23,52)),
        ]
        for p1, p2 in lines:
            cv2.line(ov, p1, p2, BDY, thickness)
        cv2.addWeighted(ov, alpha_v, frame, 1 - alpha_v, 0, frame)

    # solid skeleton (thin)
    cv2.ellipse(frame, pt(0,-82), (int(16*scale/100), int(20*scale/100)), 0,0,360,BDY,2)
    for p1, p2 in [
        (pt(-5,-62), pt(-5,-50)), (pt(5,-62), pt(5,-50)),
        (pt(-5,-52), pt(-42,-46)), (pt(5,-52), pt(42,-46)),
        (pt(-42,-46), pt(-44,8)), (pt(42,-46), pt(44,8)),
        (pt(-44,8), pt(-40,50)), (pt(44,8), pt(40,50)),
        (pt(-42,-46), pt(-28,18)), (pt(42,-46), pt(28,18)),
        (pt(-28,18), pt(-35,52)), (pt(28,18), pt(35,52)),
        (pt(-35,52), pt(35,52)),
        (pt(-23,52), pt(-21,100)), (pt(23,52), pt(21,100)),
        (pt(-35,52), pt(-23,52)), (pt(35,52), pt(23,52)),
    ]:
        cv2.line(frame, p1, p2, BDY, 2)

    # right-side annotations with values
    ann_x = int(cx + 55 * scale / 100)
    for y_body, lbl, key in [(-46,'Shoulder','shoulder_width'),
                               (-10,'Chest',   'chest'),
                               ( 18,'Waist',   'waist')]:
        p = pt(42, y_body)
        cv2.line(frame, p, (ann_x, p[1]), ANN, 1)
        cv2.circle(frame, p, 3, ANN, -1)
        v = _val(key)
        display = f'{lbl}  {v}cm' if v else lbl
        _text(frame, display, (ann_x+4, p[1]+5), ANN, 0.34, 1)

    # left-side annotations (shirt + sleeve)
    lx = int(cx - 54 * scale / 100)
    p_neck = pt(-5,-52); p_hip = pt(-28,18)
    cv2.line(frame, (lx, p_neck[1]), (lx, p_hip[1]), ANN, 1)
    cv2.circle(frame, (lx, p_neck[1]), 3, ANN, -1)
    cv2.circle(frame, (lx, p_hip[1]), 3, ANN, -1)
    mid = (p_neck[1] + p_hip[1]) // 2
    sv = _val('shirt_length')
    _text(frame, f'Shirt {sv}cm' if sv else 'Shirt', (lx-52, mid+5), ANN, 0.34, 1)

    slx = int(cx - 62 * scale / 100)
    sly = (pt(-42,-46)[1] + pt(-40,50)[1]) // 2
    cv2.line(frame, pt(-42,-46), pt(-40,50), ANN, 1)
    cv2.circle(frame, pt(-40,50), 3, ANN, -1)
    slv = _val('sleeve_length')
    _text(frame, f'Sleeve {slv}cm' if slv else 'Sleeve', (slx-58, sly+5), LBL, 0.32, 1)


# ── Human body outline for scan guide ────────────────────────────────────────
def _guide_silhouette(frame, zx, zy, zw, zh, accent, view_name, match_score):
    """
    Draw a human-shaped outline guide over the camera feed.
    Pulses with a soft glow when pose matches.
    """
    cx = zx + zw // 2

    def p(fx, fy):
        return (int(cx + fx * zw * 0.46), int(zy + fy * zh))

    thick = 2

    # Glow halo when pose is good (pulsing at ~1 Hz)
    if match_score >= 0.55:
        intensity = (match_score - 0.55) / 0.45
        pulse     = 0.55 + 0.45 * math.sin(time.time() * math.pi * 2.0)
        for g in range(3):
            a = intensity * pulse * (0.18 - g * 0.05)
            if a > 0:
                exp = (g + 1) * 7
                _alpha(frame, zx - exp, zy - exp,
                       zx + zw + exp, zy + zh + exp, 26, accent, a)

    # Head
    head_c = p(0, 0.09)
    head_r = (max(4, int(zw * 0.13)), max(4, int(zh * 0.082)))
    cv2.ellipse(frame, head_c, head_r, 0, 0, 360, accent, thick)

    # Neck
    cv2.line(frame, p(-0.05, 0.166), p(-0.05, 0.195), accent, thick)
    cv2.line(frame, p( 0.05, 0.166), p( 0.05, 0.195), accent, thick)

    if view_name in ('FRONT', 'BACK'):
        body = np.array([
            p( 0.08, 0.20), p( 0.27, 0.22), p( 0.30, 0.25),
            p( 0.30, 0.44), p( 0.26, 0.63), p( 0.19, 0.65), p( 0.10, 0.72),
            p( 0.00, 0.73),
            p(-0.10, 0.72), p(-0.19, 0.65), p(-0.26, 0.63),
            p(-0.30, 0.44), p(-0.30, 0.25), p(-0.27, 0.22), p(-0.08, 0.20),
        ], np.int32)
        cv2.polylines(frame, [body], True, accent, thick)

        # Waist guide — three subtle dots
        wy = p(0, 0.44)[1]
        for dx in (-10, 0, 10):
            cv2.circle(frame, (cx + dx, wy), 2,
                       tuple(max(0, c - 80) for c in accent), -1)

        # Arm-out arrows (FRONT only)
        if view_name == 'FRONT':
            cv2.arrowedLine(frame, p(0.30, 0.36), p(0.46, 0.36),
                            accent, 1, tipLength=0.42)
            cv2.arrowedLine(frame, p(-0.30, 0.36), p(-0.46, 0.36),
                            accent, 1, tipLength=0.42)

    else:
        # Side profile — front of body faces left for LEFT view, right for RIGHT
        fd = -1 if view_name == 'LEFT' else 1
        profile = np.array([
            p( fd * 0.07, 0.20),
            p(-fd * 0.08, 0.21),
            p(-fd * 0.13, 0.32),
            p(-fd * 0.11, 0.45),
            p(-fd * 0.09, 0.59),
            p(-fd * 0.03, 0.68),
            p( fd * 0.03, 0.68),
            p( fd * 0.09, 0.59),
            p( fd * 0.10, 0.42),
            p( fd * 0.08, 0.22),
        ], np.int32)
        cv2.polylines(frame, [profile], True, accent, thick)


# ── 3-D SMPL mesh renderer (OpenCV painter's algorithm) ──────────────────────
def render_smpl_mesh(vertices, faces, panel_w, panel_h, angle_deg=20, bg=None):
    """
    Render the body mesh as a softly lit "clay" model (the look of body-scan apps).

    vertices : (N, 3) metres, Y up;  faces: (F, 3) vertex indices
    angle_deg: turn around the vertical axis for a 3/4 view
    bg       : background colour (defaults to the panel colour)
    Returns a (panel_h, panel_w, 3) BGR image.
    """
    SS = 2                                            # supersampling → smooth edges
    W, H = panel_w * SS, panel_h * SS
    canvas = np.full((H, W, 3), PANEL if bg is None else bg, np.uint8)
    V = np.asarray(vertices, np.float64).copy()
    F = np.asarray(faces, np.int64)
    if angle_deg:
        a = math.radians(angle_deg)
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        V = V @ Ry.T

    # fit into the panel
    lo, hi = V.min(0), V.max(0)
    s = min(H * 0.86 / max(hi[1] - lo[1], 1e-6), W * 0.86 / max(hi[0] - lo[0], 1e-6))
    cx, cy = W / 2, H * 0.47
    P2 = np.stack([(V[:, 0] - (lo[0] + hi[0]) / 2) * s + cx,
                   -(V[:, 1] - (lo[1] + hi[1]) / 2) * s + cy], 1)

    # soft floor shadow
    foot_y = int(-(lo[1] - (lo[1] + hi[1]) / 2) * s + cy)
    sh = canvas.copy()
    cv2.ellipse(sh, (int(cx), foot_y + 4 * SS), (int((hi[0] - lo[0]) * s * 0.28), 9 * SS), 0, 0, 360,
                tuple(max(0, int(c) - 14) for c in canvas[0, 0]), -1, cv2.LINE_AA)
    canvas = cv2.GaussianBlur(sh, (0, 0), 6 * SS)

    # smooth (vertex) normals
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

    key = np.array([0.45, 0.55, 0.70]); key /= np.linalg.norm(key)
    fill = np.array([-0.6, 0.1, 0.5]); fill /= np.linalg.norm(fill)
    view = np.array([0.0, 0.0, 1.0])
    lam = np.clip(vn @ key, 0, 1) * 0.75 + np.clip(vn @ fill, 0, 1) * 0.20 + 0.18
    rim = np.clip(1 - np.abs(vn @ view), 0, 1) ** 3 * 0.35
    half = key + view; half /= np.linalg.norm(half)
    spec = np.clip(vn @ half, 0, 1) ** 40 * 0.25
    clay = np.array([214, 204, 196], np.float64)      # BGR light warm-grey clay
    accent = np.array(GOLD, np.float64)
    vcol = np.clip(clay[None] * lam[:, None] + accent[None] * rim[:, None] + 255 * spec[:, None], 0, 255)

    # visible faces, back to front
    fz = fn[:, 2]
    vis = np.flatnonzero(fz > 0)
    order = vis[np.argsort(V[F[vis]].mean(1)[:, 2])]
    pts = np.round(P2 * 1.0).astype(np.int32)
    fcol = vcol[F[order]].mean(1).astype(np.int32)
    for f, c in zip(F[order], fcol):
        cv2.fillConvexPoly(canvas, pts[f], (int(c[0]), int(c[1]), int(c[2])), cv2.LINE_AA)
    return cv2.resize(canvas, (panel_w, panel_h), interpolation=cv2.INTER_AREA)


def _brand(frame, y_mid=26):
    """App mark + name in the header of secondary screens (covers the old logo area)."""
    _rrect(frame, 0, y_mid - 22, 160, y_mid + 22, 0, PANEL)
    _rrect(frame, 16, y_mid - 13, 42, y_mid + 13, 7, GOLD)
    _text(frame, 'T', (24, y_mid + 7), (255, 255, 255), 0.52, 3)
    _text(frame, 'TailorCam', (50, y_mid + 7), WHITE, 0.52, 3)


# ── Secondary pages: header, past measurements, reports ──────────────────────
_RPT_SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
_RPT_BINS = {'chest': [84, 92, 100, 108, 116], 'waist': [70, 78, 86, 94, 102]}


def _page_header(frame, title, buttons=(), mouse_xy=None, height=64):
    """Header for secondary screens: brand · page title · right-aligned buttons.
    buttons: [(key, label, shortcut, kind)] with kind 'primary' | 'normal' | 'danger'.
    Returns {key: rect}."""
    h, w = frame.shape[:2]
    mx, my = mouse_xy if mouse_xy else (-1, -1)
    m = height // 2
    _rrect(frame, 0, 0, w, height, 0, PANEL)
    cv2.line(frame, (0, height), (w, height), BORDER, 1)
    _rrect(frame, 24, m - 16, 56, m + 16, 8, GOLD)
    _text(frame, 'T', (34, m + 8), (255, 255, 255), 0.62, 3)
    _text(frame, 'TailorCam', (68, m + 6), GREY, TS_BODY, 2)
    sx = 68 + _text_size('TailorCam', FONT, TS_BODY, 2)[0][0] + 14
    _text(frame, '/', (sx, m + 6), DIM, TS_BODY, 1)
    _text(frame, title, (sx + 18, m + 7), WHITE, TS_TITLE, 3)
    rects, x = {}, w - 24
    for key, label, sc, kind in reversed(list(buttons)):
        bw = _text_size(label, FONT, TS_BODY, 2)[0][0] + (_text_size(sc, FONT, TS_CAP, 1)[0][0] + 26 if sc else 0) + 36
        r = (x - bw, m - 18, x, m + 18)
        _button(frame, *r, label, sc, primary=(kind == 'primary'), danger=(kind == 'danger'),
                hover=r[0] <= mx <= r[2] and r[1] <= my <= r[3])
        rects[key] = r
        x -= bw + 10
    return rects


def draw_history_screen(frame, records, scroll=0):
    """Past measurements, newest first. Returns (frame, btns): back / exit / new."""
    h, w = frame.shape[:2]
    frame[:] = DARK
    btns = _page_header(frame, 'Past measurements', [('new', 'New scan', 'N', 'primary'), ('back', 'Home', 'H', 'normal'),
                                                      ('exit', 'Exit', 'X', 'danger')])
    P = 24
    recs = list(reversed(records or []))
    x0, x1, y0, y1 = P, w - P, 88, h - P
    _rrect(frame, x0, y0, x1, y1, 12, CARD)
    _rrect_border(frame, x0, y0, x1, y1, 12, BORDER, 1)
    cols = [('Customer', 0.0, None), ('Date', 0.26, None), ('Chest', 0.40, 'chest'), ('Waist', 0.50, 'waist'),
            ('Hip', 0.60, 'hip'), ('Shoulder', 0.70, 'shoulder_width'), ('Sleeve', 0.80, 'sleeve_length'),
            ('Size', 0.90, None)]
    cx = [x0 + 20 + int((x1 - x0 - 40) * f) for _, f, _ in cols]
    hy = y0 + 32
    for (lbl, _, _), c in zip(cols, cx):
        _text(frame, lbl.upper(), (c, hy), DIM, TS_CAP, 1)
    cv2.line(frame, (x0 + 12, hy + 12), (x1 - 12, hy + 12), BORDER, 1)
    row_h = 46
    fit = max(1, (y1 - hy - 20) // row_h)
    scroll = max(0, min(int(scroll), max(0, len(recs) - fit)))
    if not recs:
        msg = 'No measurements saved yet'
        _text(frame, msg, ((x0 + x1 - _text_size(msg, FONT, TS_BODY, 1)[0][0]) // 2, hy + 80), GREY, TS_BODY, 1)
    for i, rec in enumerate(recs[scroll:scroll + fit]):
        ry = hy + 16 + i * row_h
        mid = ry + row_h // 2 + 6
        name = rec.get('name') or rec.get('customer_name') or 'Customer'
        _avatar_circle(frame, cx[0] + 13, ry + row_h // 2, 13, name)
        _text(frame, name[:26], (cx[0] + 34, mid), WHITE, TS_BODY, 1)
        _text(frame, str(rec.get('date', rec.get('scan_date', '')))[:10], (cx[1], mid), GREY, TS_LABEL, 1)
        for (lbl, _, key), c in zip(cols[2:7], cx[2:7]):
            v = rec.get(key)
            _text(frame, f'{float(v):.1f}' if isinstance(v, (int, float)) else '—', (c, mid), WHITE if v else DIM, TS_BODY, 2)
        sz = _size_label('chest', rec.get('chest'))
        if sz:
            sw = _text_size(sz, FONT, TS_CAP, 2)[0][0]
            _rrect(frame, cx[7], ry + 12, cx[7] + sw + 16, ry + row_h - 12, 6, CARD_HI)
            _text(frame, sz, (cx[7] + 8, mid - 1), WHITE, TS_CAP, 2)
        cv2.line(frame, (x0 + 16, ry + row_h), (x1 - 16, ry + row_h), BORDER, 1)
    if len(recs) > fit:
        t = f'{scroll + 1}–{min(len(recs), scroll + fit)} of {len(recs)}  ·  scroll for more'
        _text(frame, t, (x1 - 20 - _text_size(t, FONT, TS_CAP, 1)[0][0], y0 - 8), DIM, TS_CAP, 1)
    return frame, btns


def draw_reports_screen(frame, records=None, t=0.0):
    """Shop statistics: totals, size distributions and recent activity."""
    h, w = frame.shape[:2]
    frame[:] = DARK
    _page_header(frame, 'Reports', [('back', 'Home', 'H', 'normal')])
    P, gap = 24, 16
    recs = list(reversed(records or []))
    today = datetime.date.today().isoformat()
    vals = {k: [r[k] for r in recs if isinstance(r.get(k), (int, float))] for k in ('chest', 'waist', 'shoulder_width')}
    avg = lambda k: f'{np.mean(vals[k]):.1f} cm' if vals[k] else '—'
    tiles = [('Customers', str(len({(r.get('name') or r.get('customer_name')) for r in recs}))),
             ('Scans today', str(sum(1 for r in recs if str(r.get('date', '')).startswith(today)))),
             ('Average chest', avg('chest')), ('Average waist', avg('waist')), ('Average shoulder', avg('shoulder_width'))]
    tw = (w - 2 * P - gap * (len(tiles) - 1)) // len(tiles)
    for i, (lbl, v) in enumerate(tiles):
        x = P + i * (tw + gap)
        _rrect(frame, x, 88, x + tw, 172, 12, CARD)
        _rrect_border(frame, x, 88, x + tw, 172, 12, BORDER, 1)
        _text(frame, lbl, (x + 18, 114), GREY, TS_LABEL, 1)
        _text(frame, v, (x + 18, 152), WHITE, 0.84, 3)
    # size distributions
    cy0, cy1 = 196, 196 + 220
    cw = (w - 2 * P - gap) // 2
    for j, (key, title) in enumerate((('chest', 'Chest sizes'), ('waist', 'Waist sizes'))):
        x = P + j * (cw + gap)
        _rrect(frame, x, cy0, x + cw, cy1, 12, CARD)
        _rrect_border(frame, x, cy0, x + cw, cy1, 12, BORDER, 1)
        _text(frame, title, (x + 18, cy0 + 32), WHITE, TS_TITLE, 2)
        counts = [0] * len(_RPT_SIZES)
        for v in vals[key]:
            counts[int(np.digitize(v, _RPT_BINS[key]))] += 1
        mx = max(counts) or 1
        bw = (cw - 60) // len(_RPT_SIZES)
        for k, (sz, c) in enumerate(zip(_RPT_SIZES, counts)):
            bx = x + 30 + k * bw
            bh = int((cy1 - cy0 - 100) * c / mx)
            _rrect(frame, bx + 8, cy1 - 36 - max(bh, 3), bx + bw - 8, cy1 - 36, 5, GOLD if c else CARD_HI)
            if c:
                _text(frame, str(c), (bx + bw // 2 - 4, cy1 - 44 - bh), WHITE, TS_CAP, 2)
            _text(frame, sz, (bx + bw // 2 - _text_size(sz, FONT, TS_CAP, 1)[0][0] // 2, cy1 - 14), GREY, TS_CAP, 1)
    # recent activity
    ay = cy1 + 20
    _rrect(frame, P, ay, w - P, h - P, 12, CARD)
    _rrect_border(frame, P, ay, w - P, h - P, 12, BORDER, 1)
    _text(frame, 'Recent activity', (P + 18, ay + 32), WHITE, TS_TITLE, 2)
    for i, r in enumerate(recs[: max(0, (h - P - ay - 50) // 36)]):
        ry = ay + 48 + i * 36
        name = r.get('name') or r.get('customer_name') or 'Customer'
        _text(frame, name[:28], (P + 18, ry + 22), WHITE, TS_BODY, 1)
        parts = [f"{lbl} {r[k]:.1f}" for lbl, k in (('chest', 'chest'), ('waist', 'waist')) if isinstance(r.get(k), (int, float))]
        _text(frame, '  ·  '.join(parts), (P + int((w - 2 * P) * 0.35), ry + 22), GREY, TS_LABEL, 1)
        d = str(r.get('date', ''))[:10]
        _text(frame, d, (w - P - 18 - _text_size(d, FONT, TS_LABEL, 1)[0][0], ry + 22), DIM, TS_LABEL, 1)
    return frame


# ── Modal cards: toast, turn, processing, customer details ──────────────────
def _backdrop(frame, dim=0.72):
    """Blur + darken whatever is behind a modal card."""
    _ui.bake(frame)
    small = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
    small = cv2.GaussianBlur(small, (0, 0), 3)
    frame[:] = cv2.resize(small, (frame.shape[1], frame.shape[0]))
    cv2.addWeighted(frame, 1 - dim, np.full_like(frame, DARK), dim, 0, frame)


def _modal_card(frame, cw, ch):
    h, w = frame.shape[:2]
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    _rrect(frame, x0, y0, x0 + cw, y0 + ch, 16, PANEL)
    _rrect_border(frame, x0, y0, x0 + cw, y0 + ch, 16, BORDER, 1)
    return x0, y0, x0 + cw, y0 + ch


def _hint_row(frame, x, y, items):
    """Keyboard hints:  [Tab] next field   [Enter] continue ..."""
    for key, lbl in items:
        x0_ = x
        x += _kbd(frame, x, y - 15, key) + 6
        _text(frame, lbl, (x, y), GREY, TS_CAP, 1)
        x += _text_size(lbl, FONT, TS_CAP, 1)[0][0] + 16
        _register((x0_ - 4, y - 22, x - 8, y + 10), key.split()[0])
    return x


def draw_toast(frame, message, color=GREEN, position='top-right'):
    """Notification card (top centre): dark card, coloured status dot, white text."""
    h, w = frame.shape[:2]
    tw = _text_size(message, FONT, TS_BODY, 2)[0][0]
    tw_ = min(tw + 64, w - 40)
    x0 = (w - tw_) // 2 if position != 'top-right' else w - tw_ - 20
    y0 = 20
    _rrect(frame, x0 + 2, y0 + 4, x0 + tw_ + 2, y0 + 52, 12, (8, 6, 4))      # soft shadow
    _rrect(frame, x0, y0, x0 + tw_, y0 + 48, 12, CARD_HI)
    _rrect_border(frame, x0, y0, x0 + tw_, y0 + 48, 12, BORDER, 1)
    cv2.circle(frame, (x0 + 22, y0 + 24), 6, color, -1, cv2.LINE_AA)
    _text(frame, message, (x0 + 40, y0 + 30), WHITE, TS_BODY, 2)
    return frame


def draw_turn_instruction(frame, next_view_name, views_done, total_views=2):
    """Between views: 'Front captured — now turn to your left'."""
    h, w = frame.shape[:2]
    _backdrop(frame)
    x0, y0, x1, y1 = _modal_card(frame, 560, 300)
    names = {'FRONT': 'Front', 'LEFT': 'Side', 'RIGHT': 'Right side', 'BACK': 'Back'}
    order = ['FRONT', 'LEFT', 'RIGHT', 'BACK'][:total_views]
    # stepper
    sx = x0 + 40
    for i, v in enumerate(order):
        done, cur = i < views_done, i == views_done
        cx = sx + i * 150
        cv2.circle(frame, (cx, y0 + 46), 14, GREEN if done else (GOLD if cur else CARD_HI), -1, cv2.LINE_AA)
        if done:
            cv2.line(frame, (cx - 6, y0 + 46), (cx - 2, y0 + 51), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (cx - 2, y0 + 51), (cx + 7, y0 + 41), (255, 255, 255), 2, cv2.LINE_AA)
        else:
            _text(frame, str(i + 1), (cx - 4, y0 + 52), (255, 255, 255), TS_CAP, 2)
        _text(frame, names[v], (cx + 22, y0 + 52), WHITE if (done or cur) else GREY, TS_LABEL, 2)
        if i < len(order) - 1:
            cv2.line(frame, (cx + 22 + 60, y0 + 46), (cx + 136, y0 + 46), BORDER, 2, cv2.LINE_AA)
    # arrow + text
    ax, ay = x0 + 90, y0 + 170
    cv2.circle(frame, (ax, ay), 46, CARD, -1, cv2.LINE_AA)
    d = -1 if next_view_name in ('LEFT', 'BACK') else 1
    cv2.arrowedLine(frame, (ax - 24 * d, ay), (ax + 24 * d, ay), GOLD, 5, cv2.LINE_AA, tipLength=0.45)
    title = {'LEFT': 'Turn to your left', 'RIGHT': 'Turn to your right', 'BACK': 'Turn your back to the camera'}.get(
        next_view_name, 'Get ready')
    _text(frame, f'{names.get(order[views_done - 1], "View")} photo captured', (x0 + 160, y0 + 148), GREEN, TS_LABEL, 2)
    _text(frame, title, (x0 + 160, y0 + 180), WHITE, TS_H1, 3)
    _text(frame, 'Stand straight, arms relaxed, whole body in view', (x0 + 160, y0 + 206), GREY, TS_BODY, 1)
    _text(frame, 'The next photo is taken automatically', (x0 + 40, y1 - 30), DIM, TS_CAP, 1)
    return frame


def draw_processing_screen(frame):
    """Shown while the 3D body models measure (about 10 s)."""
    h, w = frame.shape[:2]
    _backdrop(frame, 0.8)
    x0, y0, x1, y1 = _modal_card(frame, 460, 250)
    cx, cy = (x0 + x1) // 2, y0 + 70
    t = time.time()
    cv2.circle(frame, (cx, cy), 26, CARD_HI, 4, cv2.LINE_AA)
    a0 = int(t * 300) % 360
    cv2.ellipse(frame, (cx, cy), (26, 26), 0, a0, a0 + 100, GOLD, 4, cv2.LINE_AA)
    for txt, y, col, sc, th in (('Measuring the body', y0 + 134, WHITE, TS_TITLE, 3),
                                ('Building the 3D body and combining 6 models', y0 + 164, GREY, TS_BODY, 1),
                                ('This takes about 10 seconds', y0 + 204, DIM, TS_CAP, 1)):
        _text(frame, txt, (cx - _text_size(txt, FONT, sc, th)[0][0] // 2, y), col, sc, th)
    return frame


def draw_weight_age_overlay(frame, weight_text, age_text, active_field='weight',
                            gender='unspecified', warning='', height_text='', name_text=None):
    """
    Customer details form before a scan: height, weight, age and gender.
    active_field: 'height' | 'weight' | 'age'.  warning: optional message.
    """
    _backdrop(frame)
    has_name = name_text is not None
    x0, y0, x1, y1 = _modal_card(frame, 640, 360 + (88 if has_name else 0))
    P = 32
    _text(frame, 'Customer details', (x0 + P, y0 + 52), WHITE, TS_H1, 3)
    _text(frame, 'Real height and weight make every measurement accurate', (x0 + P, y0 + 80), GREY, TS_BODY, 1)
    fy = y0 + 112
    if has_name:                                   # profile name the scan is saved under
        act = active_field == 'name'
        _text(frame, 'Name', (x0 + P, fy), WHITE if act else GREY, TS_LABEL, 2)
        _text(frame, '  required', (x0 + P + _text_size('Name', FONT, TS_LABEL, 2)[0][0], fy), DIM, TS_CAP, 1)
        _rrect(frame, x0 + P, fy + 10, x1 - P, fy + 58, 10, CARD)
        _rrect_border(frame, x0 + P, fy + 10, x1 - P, fy + 58, 10, GOLD if act else BORDER, 2 if act else 1)
        _register((x0 + P, fy + 10, x1 - P, fy + 58), 'focus:name')
        shown = name_text + ('|' if act and int(time.time() * 2) % 2 == 0 else '')
        _text(frame, shown if (name_text or act) else 'Customer name', (x0 + P + 14, fy + 44),
              WHITE if name_text else DIM, TS_TITLE, 3 if name_text else 1)
        fy += 88
    fw = (x1 - x0 - 2 * P - 2 * 16) // 3
    for i, (field, label, unit, val) in enumerate((('height', 'Height', 'cm', height_text),
                                                   ('weight', 'Weight', 'kg', weight_text),
                                                   ('age', 'Age', 'years', age_text))):
        fx = x0 + P + i * (fw + 16)
        act = active_field == field
        req = '' if field == 'age' else '  required'
        _text(frame, label, (fx, fy), WHITE if act else GREY, TS_LABEL, 2)
        if req:
            _text(frame, req, (fx + _text_size(label, FONT, TS_LABEL, 2)[0][0], fy), DIM, TS_CAP, 1)
        _register((fx, fy + 10, fx + fw, fy + 58), f'focus:{field}')
        _rrect(frame, fx, fy + 10, fx + fw, fy + 58, 10, CARD)
        _rrect_border(frame, fx, fy + 10, fx + fw, fy + 58, 10, GOLD if act else BORDER, 2 if act else 1)
        shown = val + ('|' if act and int(time.time() * 2) % 2 == 0 else '')
        _text(frame, shown if val or act else '—', (fx + 14, fy + 44), WHITE if val else DIM, TS_TITLE, 3)
        _text(frame, unit, (fx + fw - 14 - _text_size(unit, FONT, TS_CAP, 1)[0][0], fy + 42), DIM, TS_CAP, 1)
    gy = fy + 92
    _text(frame, 'Gender', (x0 + P, gy), GREY, TS_LABEL, 2)
    opts = (('male', 'Male', 'M'), ('female', 'Female', 'F'), ('unspecified', 'Not given', 'U'))
    sw = (x1 - x0 - 2 * P) // 3
    _rrect(frame, x0 + P, gy + 10, x1 - P, gy + 52, 10, CARD)
    for i, (g, lbl, key) in enumerate(opts):
        sx = x0 + P + i * sw
        _register((sx, gy + 10, sx + sw, gy + 52), key)
        on = gender == g
        if on:
            _rrect(frame, sx + 4, gy + 14, sx + sw - 4, gy + 48, 8, GOLD)
        tw = _text_size(lbl, FONT, TS_BODY, 2)[0][0]
        _text(frame, lbl, (sx + (sw - tw) // 2 - 10, gy + 37), (255, 255, 255) if on else GREY, TS_BODY, 2)
        _text(frame, key, (sx + (sw + tw) // 2, gy + 37), (230, 230, 255) if on else DIM, TS_CAP, 1)
    if warning:
        cv2.circle(frame, (x0 + P + 5, y1 - 66), 5, AMBER, -1, cv2.LINE_AA)
        _text(frame, warning, (x0 + P + 18, y1 - 60), AMBER, TS_CAP, 2)
    _hint_row(frame, x0 + P, y1 - 24, (('Tab', 'next field'),) +
              ((('M F U', 'gender'),) if active_field != 'name' else ()) +
              (('Enter', 'continue'), ('Esc', 'skip weight')))
    return frame


# ── Input pop-ups + tailor feedback ──────────────────────────────────────────
def _input_card(frame, title, subtitle, text, hint, accent=GOLD, unit=''):
    """Single-field modal (height, edit value, name, folder)."""
    _backdrop(frame)
    x0, y0, x1, y1 = _modal_card(frame, 560, 230)
    P = 28
    _text(frame, title, (x0 + P, y0 + 46), WHITE, TS_TITLE + 0.06, 3)
    _text(frame, subtitle, (x0 + P, y0 + 74), GREY, TS_BODY, 1)
    bx0, by0, bx1, by1 = x0 + P, y0 + 92, x1 - P, y0 + 146
    _rrect(frame, bx0, by0, bx1, by1, 10, CARD)
    _rrect_border(frame, bx0, by0, bx1, by1, 10, accent, 2)
    cur = '|' if int(time.time() * 2) % 2 == 0 else ''
    _text(frame, (text or '') + cur, (bx0 + 16, by1 - 18), WHITE, TS_TITLE + 0.06, 3)
    if unit:
        _text(frame, unit, (bx1 - 16 - _text_size(unit, FONT, TS_LABEL, 1)[0][0], by1 - 20), DIM, TS_LABEL, 1)
    items = []
    for part in hint.split('   '):
        if '=' in part:
            k, v = part.split('=', 1)
            items.append((k.strip(), v.strip()))
    _hint_row(frame, x0 + P, y1 - 26, items)
    return frame


def draw_height_input_overlay(frame, text):
    return _input_card(frame, 'Customer height',
                       'Type the real height (from a tape or ID card)',
                       text, 'Enter = save   Esc = cancel', GOLD, 'cm')


def draw_measurement_edit_overlay(frame, _field_key, field_label, current_val, input_text):
    """Overlay to manually edit a single measurement value."""
    cur = f'{current_val:.1f} cm' if isinstance(current_val, (int, float)) else 'not measured'
    return _input_card(frame, f"Edit {field_label.replace(' Circ.', '').lower()}",
                       f'Measured {cur} — type the corrected value', input_text,
                       'Enter = save   Esc = cancel', GOLD, 'cm')


def draw_name_input_overlay(frame, text):
    return _input_card(frame, 'Save PDF report', 'Customer name for the report', text,
                       'Enter = save   Esc = cancel', GOLD)


def draw_folder_input_overlay(frame, text):
    return _input_card(frame, 'Save folder', 'Folder path (or choose it in the window that opens)',
                       text, 'Enter = confirm   Esc = cancel', GOLD)


_FEEDBACK_LABELS = {'chest': 'Chest', 'waist': 'Waist', 'hip': 'Hip', 'shoulder_width': 'Shoulder',
                    'sleeve_length': 'Sleeve', 'back_length': 'Back length', 'shirt_length': 'Shirt length',
                    'neck': 'Neck'}


def draw_feedback_overlay(frame, final_m, keys, field_idx, vals):
    """
    Tailor feedback: the app's value next to the real tape value for each
    measurement. Saved values retrain the app for this shop.
    """
    _backdrop(frame)
    h = frame.shape[0]
    rows = len(keys)
    x0, y0, x1, y1 = _modal_card(frame, 600, min(h - 40, 150 + rows * 46))
    P = 28
    _text(frame, 'Tape measurements', (x0 + P, y0 + 46), WHITE, TS_TITLE + 0.06, 3)
    _text(frame, 'Type what the tape shows — the app learns from every customer', (x0 + P, y0 + 72), GREY, TS_BODY, 1)
    c_app, c_tape = x0 + 260, x0 + 400
    _text(frame, 'APP', (c_app, y0 + 100), DIM, TS_CAP, 1)
    _text(frame, 'TAPE (cm)', (c_tape, y0 + 100), DIM, TS_CAP, 1)
    for i, k in enumerate(keys):
        ry = y0 + 110 + i * 46
        if ry + 46 > y1 - 40:
            break
        act = i == field_idx
        if act:
            _rrect(frame, x0 + 12, ry + 2, x1 - 12, ry + 42, 9, CARD_HI)
        mid = ry + 28
        _text(frame, _FEEDBACK_LABELS.get(k, k.replace('_', ' ').title()), (x0 + P, mid), WHITE if act else GREY, TS_BODY, 2 if act else 1)
        v = (final_m or {}).get(k)
        _text(frame, f'{v:.1f}' if isinstance(v, (int, float)) else '—', (c_app, mid), GREY, TS_BODY, 1)
        bx1 = x1 - P
        _rrect(frame, c_tape - 10, ry + 7, bx1, ry + 37, 7, CARD)
        _rrect_border(frame, c_tape - 10, ry + 7, bx1, ry + 37, 7, GOLD if act else BORDER, 2 if act else 1)
        t = vals.get(k, '')
        cur = '|' if act and int(time.time() * 2) % 2 == 0 else ''
        _text(frame, (t + cur) if (t or act) else 'skip', (c_tape, mid), WHITE if t else DIM, TS_BODY, 2)
    _hint_row(frame, x0 + P, y1 - 22, (('Tab', 'next'), ('K', 'previous'), ('Enter', 'save all'), ('Esc', 'cancel')))
    return frame


# ── Results tabs: history + size guide ───────────────────────────────────────
def _draw_history_tab(frame, px, y, total_w, total_h, history_records, current_m, card_rects):
    """
    Earlier scans of this customer, newest first. Each row is clickable → the
    caller loads that scan. Adds 'hist_N' rects and '_hist_records' to card_rects.
    """
    x0, x1 = px + 20, px + total_w - 20
    y1 = y + total_h
    _rrect(frame, x0, y, x1, y1, 12, CARD)
    _rrect_border(frame, x0, y, x1, y1, 12, BORDER, 1)
    _text(frame, 'Earlier scans', (x0 + 20, y + 34), WHITE, TS_TITLE, 2)
    _text(frame, 'Click a scan to load it into Measurements', (x0 + 20, y + 56), GREY, TS_CAP, 1)
    cols = [('Date', 0.0), ('Chest', 0.26), ('Waist', 0.40), ('Shoulder', 0.54), ('Sleeve', 0.70), ('', 0.84)]
    cx = [x0 + 20 + int((x1 - x0 - 40) * f) for _, f in cols]
    hy = y + 88
    for (lbl, _), c in zip(cols, cx):
        _text(frame, lbl.upper(), (c, hy), DIM, TS_CAP, 1)
    cv2.line(frame, (x0 + 12, hy + 10), (x1 - 12, hy + 10), BORDER, 1)
    recs = history_records or []
    if not recs:
        msg = 'No earlier scans for this customer yet'
        _text(frame, msg, ((x0 + x1 - _text_size(msg, FONT, TS_BODY, 1)[0][0]) // 2, hy + 70), GREY, TS_BODY, 1)
    row_h = 44
    for i, rec in enumerate(recs):
        ry = hy + 14 + i * row_h
        if ry + row_h > y1 - 8:
            break
        card_rects[f'hist_{i}'] = (x0, ry, x1, ry + row_h)
        same = current_m and all(current_m.get(k) == rec.get(k) for k in ('chest', 'waist'))
        if same:
            _rrect(frame, x0 + 6, ry + 2, x1 - 6, ry + row_h - 2, 8, CARD_HI)
        mid = ry + row_h // 2 + 6
        _text(frame, str(rec.get('scan_date', rec.get('date', '')))[:10], (cx[0], mid), WHITE, TS_BODY, 1)
        for c, k in zip(cx[1:5], ('chest', 'waist', 'shoulder_width', 'sleeve_length')):
            v = rec.get(k)
            _text(frame, f'{v:.1f}' if isinstance(v, (int, float)) else '—', (c, mid), WHITE if v else DIM, TS_BODY, 2)
        _text(frame, 'current' if same else 'load', (cx[5], mid), GREEN if same else GOLD, TS_CAP, 2)
        if i < len(recs) - 1:
            cv2.line(frame, (x0 + 16, ry + row_h), (x1 - 16, ry + row_h), BORDER, 1)
    card_rects['_hist_records'] = history_records


def _draw_size_guide_tab(frame, px, y, total_w, total_h, measurements, country='BD'):
    """
    Ready-made sizes for every measurement in the chosen country's system
    (C cycles the country). Body measurement + maker's ease = garment size.
    """
    x0, x1 = px + 20, px + total_w - 20
    y1 = y + total_h
    _rrect(frame, x0, y, x1, y1, 12, CARD)
    _rrect_border(frame, x0, y, x1, y1, 12, BORDER, 1)
    info = _COUNTRY_SIZES.get(country, {})
    _text(frame, f"Sizes · {info.get('label', country)}", (x0 + 20, y + 34), WHITE, TS_TITLE, 2)
    if info.get('ease_note'):
        _text(frame, info['ease_note'], (x0 + 20, y + 56), GREY, TS_CAP, 1)
    # country chips (C = next)
    cx = x1 - 20
    for ck in reversed(COUNTRY_KEYS):
        tw = _text_size(ck, FONT, TS_CAP, 2)[0][0] + 16
        cx -= tw + 4
        on = ck == country
        _rrect(frame, cx, y + 18, cx + tw, y + 40, 8, GOLD if on else CARD_HI)
        _text(frame, ck, (cx + 8, y + 34), (255, 255, 255) if on else GREY, TS_CAP, 2)
    _kbd(frame, cx - 30, y + 19, 'C')
    cols = [('Measurement', 0.0), ('Body', 0.36), ('Garment', 0.54), ('Size', 0.74)]
    cxs = [x0 + 20 + int((x1 - x0 - 40) * f) for _, f in cols]
    hy = y + 88
    for (lbl, _), c in zip(cols, cxs):
        _text(frame, lbl.upper(), (c, hy), DIM, TS_CAP, 1)
    cv2.line(frame, (x0 + 12, hy + 10), (x1 - 12, hy + 10), BORDER, 1)
    row_h = min(44, (y1 - hy - 70) // len(MEASUREMENT_ORDER))
    for i, (key, label) in enumerate(MEASUREMENT_ORDER):
        ry = hy + 14 + i * row_h
        mid = ry + row_h // 2 + 6
        val = (measurements or {}).get(key)
        _text(frame, label.replace(' Circ.', ''), (cxs[0], mid), GREY, TS_BODY, 1)
        if val is None:
            _text(frame, '—', (cxs[1], mid), DIM, TS_BODY, 1)
        else:
            dv, unit = _display_val(val)
            _text(frame, f'{dv} {unit}', (cxs[1], mid), WHITE, TS_BODY, 2)
            g = _garment_cm(key, val)
            _text(frame, f'~{g:.0f} cm' if g and key in ('chest', 'waist', 'hip', 'neck') else '—', (cxs[2], mid),
                  GREY if g else DIM, TS_BODY, 1)
            sz = _country_size(country, key if key in ('chest', 'waist', 'neck') else '', val) or _iso_letter(key, val)
            if sz:
                sw = _text_size(sz, FONT, TS_BODY, 3)[0][0]
                _rrect(frame, cxs[3] - 8, ry + 8, cxs[3] + sw + 8, ry + row_h - 8, 7, CARD_HI)
                _text(frame, sz, (cxs[3], mid), WHITE, TS_BODY, 3)
        if i < len(MEASUREMENT_ORDER) - 1:
            cv2.line(frame, (x0 + 16, ry + row_h), (x1 - 16, ry + row_h), BORDER, 1)
    rec = _garment_recommendation(measurements or {}, country)
    if rec:
        _text(frame, f'Recommended:  {rec}', (x0 + 20, y1 - 22), GOLD, TS_LABEL, 2)


# ── Results overlay ───────────────────────────────────────────────────────────
# Gauge ranges [lo, hi] in cm for visual bar
_GAUGE_RANGES = {
    'shoulder_width': (32, 56),
    'chest':          (72, 130),
    'waist':          (58, 110),
    'back_length':    (38, 58),
    'shirt_length':   (62, 86),
    'sleeve_length':  (52, 76),
    'neck':           (30, 50),
}

_GARMENT_DISPLAY = {
    'tight_fit': 'Tight', 't_shirt': 'T-Shirt',
    'shirt': 'Shirt', 'sweater': 'Sweater', 'jacket': 'Jacket',
}


def _chip(frame, x, y, label, fg, bg=None, scale=None):
    """Small rounded tag. Returns its width."""
    scale = scale or TS_CAP
    tw = _text_size(label, FONT, scale, 2)[0][0]
    _rrect(frame, x, y, x + tw + 16, y + 22, 11, bg if bg is not None else CARD_HI)
    _text(frame, label, (x + 8, y + 16), fg, scale, 2)
    return tw + 16


def _segmented(frame, x, y, w, labels, active):
    """Tab control (segmented buttons)."""
    _rrect(frame, x, y, x + w, y + 38, 10, CARD)
    sw = (w - 8) // len(labels)
    for i, (lbl, key) in enumerate(labels):
        sx = x + 4 + i * sw
        _register((sx, y, sx + sw, y + 38), key)
        if i == active:
            _rrect(frame, sx, y + 4, sx + sw, y + 34, 8, CARD_HI)
        txt = f'{lbl}'
        tw = _text_size(txt, FONT, TS_LABEL, 2)[0][0]
        kw = _text_size(key, FONT, TS_CAP, 1)[0][0] + 12
        cx = sx + (sw - tw - kw - 8) // 2
        _text(frame, txt, (cx, y + 25), WHITE if i == active else GREY, TS_LABEL, 2)
        _rrect(frame, cx + tw + 8, y + 10, cx + tw + 8 + kw, y + 28, 5, PANEL if i == active else CARD_HI)
        _text(frame, key, (cx + tw + 14, y + 24), GREY, TS_CAP, 1)


def draw_results_overlay(frame, measurements, countdown=None, mesh_img=None,
                         refining=False, reveal_elapsed=9999.0,
                         uncertainties=None, garment_type='t_shirt',
                         posture_score=None, prev_measurements=None,
                         active_tab=0, customer_name='', profile_photo=None,
                         country='BD', edit_field=None, history_records=None,
                         tape_status=None):
    """
    Results screen. Left: 3D body + customer; right: Measurements | Size guide |
    History (keys 1/2/3); bottom: actions.
    tape_status: {key: {'status': 'camera'|'tape', 'expected_error': cm}} (model.calibration.tape_check)
    → "CAM ±x" / "TAPE ±x" tags. Returns (frame, card_rects): measurement row rects
    (click to edit) plus history row rects 'hist_0', ... from the History tab.
    """
    h, w = frame.shape[:2]
    P = 20
    px = int(w * 0.40)
    measurements = measurements or {}
    frame[:] = DARK

    # ── Left panel: who + 3D body ─────────────────────────────────────────────
    _rrect(frame, 0, 0, px, h, 0, PANEL)
    cv2.line(frame, (px, 0), (px, h), BORDER, 1)
    name = customer_name or 'Customer'
    _avatar_circle(frame, P + 18, P + 20, 18, name)
    _text(frame, name[:26], (P + 46, P + 18), WHITE, TS_TITLE, 2)
    g_label = _GARMENT_DISPLAY.get(garment_type, garment_type)
    _text(frame, f'{datetime.datetime.now():%d %b %Y}  ·  {g_label}', (P + 46, P + 38), GREY, TS_CAP, 1)
    sx = P
    if refining:
        dots = '.' * (1 + int(time.time() * 2) % 3)
        sx += _chip(frame, sx, P + 54, f'Refining{dots}', GOLD) + 8
    else:
        sx += _chip(frame, sx, P + 54, 'Scan complete', GREEN) + 8
    if tape_status:
        n_tape = sum(1 for k, _ in MEASUREMENT_ORDER if (tape_status.get(k) or {}).get('status') == 'tape')
        if n_tape:
            _chip(frame, sx, P + 54, f'Tape-check {n_tape} of {len(MEASUREMENT_ORDER)}', AMBER)

    by0, by1 = P + 92, h - 76                            # body area
    if mesh_img is not None:
        mh, mw = mesh_img.shape[:2]
        s = min((px - 2 * P) / mw, (by1 - by0) / mh)
        tw_, th_ = max(1, int(mw * s)), max(1, int(mh * s))
        x0 = (px - tw_) // 2
        img = cv2.resize(mesh_img, (tw_, th_), interpolation=cv2.INTER_AREA)
        frame[by0:by0 + th_, x0:x0 + tw_] = img
    else:
        cx, cy, sc = px // 2, (by0 + by1) // 2, int((by1 - by0) * 0.40)
        _filled_silhouette(frame, cx, cy, sc, measurements, alpha_fill=0.18)

    # bottom strip of the left panel: body type + capture quality
    btype = classify_body_type(measurements)
    _text(frame, 'Body type', (P, h - 48), DIM, TS_CAP, 1)
    _text(frame, str(btype).title(), (P, h - 26), WHITE, TS_BODY, 2)
    if posture_score is not None:
        q = max(0.0, min(1.0, float(posture_score)))
        qx = px // 2
        _text(frame, 'Capture quality', (qx, h - 48), DIM, TS_CAP, 1)
        bw = px - qx - P
        _rrect(frame, qx, h - 34, qx + bw, h - 28, 3, CARD_HI)
        _rrect(frame, qx, h - 34, qx + max(6, int(bw * q)), h - 28, 3, GREEN if q >= .75 else (AMBER if q >= .45 else RED))
        _text(frame, f'{int(q * 100)}%', (qx, h - 12), GREY, TS_CAP, 1)

    # ── Right panel: tabs ─────────────────────────────────────────────────────
    rx0, rx1 = px + P, w - P
    _segmented(frame, rx0, P, rx1 - rx0, [('Measurements', '1'), ('Size guide', '2'), ('History', '3')], active_tab)
    content_y = P + 50
    footer_y = h - 64
    content_h = footer_y - content_y - 8
    card_rects = {}

    if active_tab == 0:
        _rrect(frame, rx0, content_y, rx1, footer_y - 8, 12, CARD)
        _rrect_border(frame, rx0, content_y, rx1, footer_y - 8, 12, BORDER, 1)
        n = len(MEASUREMENT_ORDER)
        row_h = (footer_y - 8 - content_y) // n
        for i, (key, label) in enumerate(MEASUREMENT_ORDER):
            ry = content_y + i * row_h
            card_rects[key] = (rx0, ry, rx1, ry + row_h)
            if edit_field == key:
                _rrect(frame, rx0 + 4, ry + 3, rx1 - 4, ry + row_h - 3, 9, CARD_HI)
                _rrect_border(frame, rx0 + 4, ry + 3, rx1 - 4, ry + row_h - 3, 9, GOLD, 2)
            mid = ry + row_h // 2
            val = measurements.get(key)
            _text(frame, label.replace(' Circ.', ''), (rx0 + 20, mid - 6), GREY, TS_LABEL, 1)
            ts = (tape_status or {}).get(key)
            if ts:
                err = ts.get('expected_error')
                tag = ('Camera OK' if ts['status'] == 'camera' else 'Check with tape') + (f'  ±{err:.1f}' if err else '')
                fg = GREEN if ts['status'] == 'camera' else AMBER
                _chip(frame, rx0 + 20, mid + 2, tag, fg, scale=0.36)
            # value
            if val is None:
                _text(frame, 'click to enter', (rx0 + int((rx1 - rx0) * 0.42), mid + 7), DIM, TS_LABEL, 1)
            else:
                age = reveal_elapsed - i * 0.06
                shown = round(val * _ease_out(age / 0.5), 1) if 0 <= age < 0.5 and isinstance(val, (int, float)) else val
                dv, unit = _display_val(shown)
                vx = rx0 + int((rx1 - rx0) * 0.42)
                _text(frame, dv, (vx, mid + 10), WHITE, 0.86, 3)
                vw = _text_size(dv, FONT, 0.86, 3)[0][0]
                _text(frame, unit, (vx + vw + 6, mid + 10), GREY, TS_LABEL, 1)
                if prev_measurements and isinstance(prev_measurements.get(key), (int, float)):
                    d = round(val - prev_measurements[key], 1)
                    if abs(d) >= 0.2:
                        _text(frame, f'{d:+.1f} since last', (vx, mid + 28), DIM, TS_CAP, 1)
                # size
                sz = _country_size(country, key if key in ('chest', 'waist', 'neck') else '', val) or _iso_letter(key, val)
                if sz:
                    sw = _text_size(sz, FONT, TS_TITLE, 3)[0][0]
                    sx2 = rx1 - 24 - max(sw, 40)
                    _rrect(frame, sx2 - 12, mid - 18, rx1 - 16, mid + 16, 9, CARD_HI)
                    _text(frame, sz, (sx2 + (max(sw, 40) - sw) // 2 - 2, mid + 7), WHITE, TS_TITLE, 3)
                    g_cm = _garment_cm(key, val)
                    if g_cm and key in ('chest', 'waist', 'hip', 'neck'):
                        gt = f'garment ~{g_cm:.0f} cm'
                        _text(frame, gt, (sx2 - 24 - _text_size(gt, FONT, TS_CAP, 1)[0][0], mid + 6), DIM, TS_CAP, 1)
            if i < n - 1:
                cv2.line(frame, (rx0 + 16, ry + row_h), (rx1 - 16, ry + row_h), BORDER, 1)
    elif active_tab == 1:
        _draw_size_guide_tab(frame, px, content_y, w - px, content_h, measurements, country=country)
    elif active_tab == 2:
        _draw_history_tab(frame, px, content_y, w - px, content_h, history_records, measurements, card_rects)

    # ── Actions ───────────────────────────────────────────────────────────────
    acts = [('Save PDF', 'S', True), ('WhatsApp', 'W', False), ('Garment', 'G', False),
            ('Feedback', 'T', False), ('Next', 'N', False), ('Restart', 'R', False)]
    gap = 8
    bw = (rx1 - rx0 - gap * (len(acts) - 1)) // len(acts)
    for i, (lbl, key, prim) in enumerate(acts):
        x = rx0 + i * (bw + gap)
        _button(frame, x, footer_y, x + bw, footer_y + 44, lbl, key, primary=prim)
    return frame, card_rects


# ── Input overlays ─────────────────────────────────────────────────────────────
def draw_inter_scan_overlay(frame, scan_num, total_scans, partial_m):
    """Shown between scans: "Scan N/total complete, turn back for next scan." """
    h, w = frame.shape[:2]
    _alpha(frame, 0, 0, w, h, 0, DARK, 0.72)
    cw, ch = min(580, w - 60), 176
    cx0 = (w - cw) // 2;  cy0 = (h - ch) // 2 - 20
    _alpha(frame, cx0, cy0, cx0+cw, cy0+ch, 14, PANEL, 0.96)
    _rrect_border(frame, cx0, cy0, cx0+cw, cy0+ch, 14, GREEN, 2)
    # Header band
    _rrect(frame, cx0, cy0, cx0+cw, cy0+44, 14, CARD)
    _text(frame, f'SCAN  {scan_num} / {total_scans}  COMPLETE',
          (cx0+24, cy0+29), GREEN, 0.68, 2, FONT_DUP)
    remaining = total_scans - scan_num
    s = 's' if remaining > 1 else ''
    msg = f'{remaining} more scan{s} remaining  --  averaging {total_scans} scans for accuracy'
    _text(frame, msg, (cx0+16, cy0+66), GREY, 0.48, 1)
    _text(frame, 'Turn back to face the camera  --  next scan starts automatically',
          (cx0+16, cy0+96), WHITE, 0.46, 1)
    # Progress dots
    dot_r, dot_y = 8, cy0 + 130
    spacing = 28
    total_w = total_scans * (dot_r*2) + (total_scans-1) * (spacing - dot_r*2)
    dot_x0 = (w - total_w) // 2
    for i in range(total_scans):
        cx = dot_x0 + i * spacing + dot_r
        col = GREEN if i < scan_num else BORDER
        cv2.circle(frame, (cx, dot_y), dot_r, col, -1)
        if i < scan_num:
            cv2.circle(frame, (cx, dot_y), dot_r + 2, GREEN, 1)
    # Partial measurement peek
    if partial_m:
        chest    = partial_m.get('chest', '-')
        shoulder = partial_m.get('shoulder_width', '-')
        sleeve   = partial_m.get('sleeve_length', '-')
        peek = f'This scan:  shoulder {shoulder} cm    chest {chest} cm    sleeve {sleeve} cm'
        _text(frame, peek, (cx0+16, cy0+ch-10), DIM, 0.40, 1)
    return frame


def _draw_qr_badge(frame, x, y, size, measurements):
    """
    Draw a compact measurement QR-style info badge.
    Shows a simple grid-pixel pattern that encodes a visual identity
    for this scan (deterministic from the values), plus key stats.
    """
    cell = max(2, size // 14)
    grid = size // cell
    # Seed a deterministic pattern from measurement sum
    seed_v = int(sum(v for v in measurements.values() if v) * 10) & 0xFFFF
    np.random.seed(seed_v)
    pattern = np.random.randint(0, 2, (grid, grid), dtype=np.uint8)
    # Always-on corner markers (QR-like finder patterns)
    for (qr, qc) in [(0,0),(0,grid-3),(grid-3,0)]:
        pattern[qr:qr+3, qc:qc+3] = 1
        if qr+1 < grid and qc+1 < grid:
            pattern[qr+1, qc+1] = 0

    bg = (58, 32, 18)
    _rrect(frame, x, y, x + size, y + size, 4, bg)
    _rrect_border(frame, x, y, x + size, y + size, 4, GOLD, 1)
    for r in range(min(grid, 14)):
        for c in range(min(grid, 14)):
            if pattern[r, c]:
                rx1 = x + c * cell + 1
                ry1 = y + r * cell + 1
                cv2.rectangle(frame, (rx1, ry1), (rx1 + cell - 2, ry1 + cell - 2),
                              GOLD, -1)
    np.random.seed(None)   # restore randomness


# ── New UI component helpers ────────────────────────────────────────────────────

def _gauge_bar(frame, x, y, bar_w, bar_h, value, lo, hi, accent):
    """Horizontal progress bar: value within [lo..hi], filled with accent color."""
    ratio = max(0.0, min(1.0, (value - lo) / max(hi - lo, 1.0)))
    _rrect(frame, x, y, x + bar_w, y + bar_h, bar_h // 2, BORDER)
    fill_w = max(bar_h, int(bar_w * ratio))
    if fill_w > 0:
        _rrect(frame, x, y, x + fill_w, y + bar_h, bar_h // 2, accent)
    tx2 = x + fill_w
    if 0 < fill_w < bar_w:
        cv2.circle(frame, (tx2, y + bar_h // 2), bar_h // 2 + 1, WHITE, -1)
        cv2.circle(frame, (tx2, y + bar_h // 2), bar_h // 2 - 1, accent, -1)
    return ratio


def _icon_btn(frame, x, y, bw, bh, label, key_hint, col, hovered=False, active=False):
    """Pill-shaped button: small key chip + label text."""
    bg = tuple(min(255, c + 25) for c in CARD) if (hovered or active) else CARD
    _rrect(frame, x, y, x + bw, y + bh, bh // 2, bg)
    border_col = WHITE if hovered else col
    _rrect_border(frame, x, y, x + bw, y + bh, bh // 2, border_col, 2 if (hovered or active) else 1)
    kw = _text_size(key_hint, FONT, 0.28, 1)[0][0] + 8
    ky = y + (bh - 14) // 2
    _rrect(frame, x + 5, ky, x + 5 + kw, ky + 14, 4, PANEL)
    _text(frame, key_hint, (x + 9, ky + 11), col, 0.28, 1)
    lw = _text_size(label, FONT, 0.38, 1)[0][0]
    lx2 = x + 5 + kw + (bw - 10 - kw - lw) // 2
    _text(frame, label, (lx2, y + bh // 2 + 6), WHITE if (hovered or active) else col, 0.38, 1)


def _tab_bar(frame, x, y, total_w, tab_h, tabs, active_idx):
    """Horizontal tab bar. Returns list of (x1,y1,x2,y2) hit regions."""
    tab_w = total_w // len(tabs)
    _rrect(frame, x, y, x + total_w, y + tab_h, 0, PANEL)
    cv2.line(frame, (x, y + tab_h), (x + total_w, y + tab_h), BORDER, 1)
    regions = []
    for i, lbl in enumerate(tabs):
        tx0 = x + i * tab_w
        tx1 = tx0 + tab_w
        is_active = (i == active_idx)
        if is_active:
            _rrect(frame, tx0 + 2, y + 2, tx1 - 2, y + tab_h - 1, 6, CARD)
            cv2.line(frame, (tx0 + 2, y + tab_h - 1), (tx1 - 2, y + tab_h - 1), GOLD, 2)
        tw2 = _text_size(lbl, FONT, 0.38, 1)[0][0]
        c_tab = GOLD if is_active else GREY
        _text(frame, lbl, (tx0 + (tab_w - tw2) // 2, y + tab_h - 7), c_tab, 0.38, 1)
        regions.append((tx0, y, tx1, y + tab_h))
    return regions


def _filled_silhouette(frame, cx, cy, scale, measurements=None, alpha_fill=0.16):
    """Filled body silhouette with color-coded measurement zones."""
    def pt(fx, fy):
        return (int(cx + fx * scale / 100), int(cy + fy * scale / 100))

    ov = frame.copy()
    head_c = pt(0, -82)
    head_r = (int(16 * scale / 100), int(20 * scale / 100))
    if head_r[0] > 0 and head_r[1] > 0:
        cv2.ellipse(ov, head_c, head_r, 0, 0, 360, GOLD, -1)

    torso = np.array([pt(-28,-62),pt(28,-62),pt(32,-46),pt(30,-10),
                      pt(24,18),pt(-24,18),pt(-30,-10),pt(-32,-46)], np.int32)
    cv2.fillPoly(ov, [torso], GOLD)

    hip = np.array([pt(-24,18),pt(24,18),pt(28,52),pt(-28,52)], np.int32)
    cv2.fillPoly(ov, [hip], GREEN)

    for side in [-1, 1]:
        arm = np.array([pt(side*32,-46),pt(side*44,-40),pt(side*46,8),
                        pt(side*40,50),pt(side*36,50),pt(side*40,8),
                        pt(side*38,-42)], np.int32)
        cv2.fillPoly(ov, [arm], TEAL)

    for side in [-1, 1]:
        leg = np.array([pt(side*6,52),pt(side*28,52),
                        pt(side*24,100),pt(side*12,100)], np.int32)
        cv2.fillPoly(ov, [leg], DIM)

    cv2.addWeighted(ov, alpha_fill, frame, 1.0 - alpha_fill, 0, frame)

    if head_r[0] > 0 and head_r[1] > 0:
        cv2.ellipse(frame, head_c, head_r, 0, 0, 360, GOLD, 1)
    cv2.polylines(frame, [torso], True, GOLD, 1)
    cv2.polylines(frame, [hip], True, GREEN, 1)

    if measurements:
        ann_x = int(cx + 60 * scale / 100)
        for y_b, lbl, key in [(-46,'Shoulder','shoulder_width'),
                               (-10,'Chest','chest'),
                               (18,'Waist','waist')]:
            p = pt(32, y_b)
            cv2.line(frame, p, (ann_x, p[1]), GOLD, 1)
            cv2.circle(frame, p, 3, GOLD, -1)
            v = measurements.get(key)
            txt = f'{lbl} {v:.0f}cm' if v else lbl
            _text(frame, txt, (ann_x + 4, p[1] + 5), GOLD, 0.28, 1)


# ── Inch / cm toggle ──────────────────────────────────────────────────────────
USE_INCHES = False   # flipped by main.py on I key; persists in config

def _cm_to_in(v):
    return round(v / 2.54, 1) if v is not None else None

def _display_val(v_cm, decimals=1):
    """Return (value_str, unit_str) respecting USE_INCHES toggle."""
    if v_cm is None:
        return '--', ('in' if USE_INCHES else 'cm')
    v = _cm_to_in(v_cm) if USE_INCHES else v_cm
    fmt = f'{v:.{decimals}f}'
    return fmt, ('in' if USE_INCHES else 'cm')


# ── Body type classification ──────────────────────────────────────────────────
_BODY_TYPE_LABELS = {
    'hourglass':  'Hourglass',
    'pear':       'Pear',
    'apple':      'Apple',
    'rectangle':  'Rectangle',
    'inverted':   'Inverted Triangle',
    'unknown':    'Unknown',
}
_BODY_TYPE_COLORS = {
    'hourglass': (0, 165, 240),   # GOLD
    'pear':      (50, 195, 40),   # GREEN
    'apple':     (45, 42, 215),   # RED
    'rectangle': (160, 120, 0),   # TEAL
    'inverted':  (0, 165, 240),   # GOLD
    'unknown':   (52, 54, 68),    # DIM
}

def classify_body_type(measurements):
    """
    Classify body type from chest / waist / hip / shoulder ratios.
    Returns one of: hourglass | pear | apple | rectangle | inverted | unknown
    """
    chest    = measurements.get('chest')
    waist    = measurements.get('waist')
    hip      = measurements.get('hip') or measurements.get('chest')
    shoulder = measurements.get('shoulder_width')
    if not chest or not waist:
        return 'unknown'
    hip = hip or chest * 1.02
    try:
        w_h   = waist  / max(hip,   1)
        w_c   = waist  / max(chest, 1)
        sh_h  = (shoulder * 2.3) / max(hip, 1) if shoulder else 1.0  # est. bust from shoulder
        is_defined_waist = w_h < 0.80 and w_c < 0.82
        if is_defined_waist and abs(chest - hip) < 6:
            return 'hourglass'
        elif hip > chest + 5 and w_h < 0.78:
            return 'pear'
        elif chest > hip + 4 or sh_h > 1.06:
            return 'inverted'
        elif waist > chest - 6 and waist > hip - 6:
            return 'apple'
        else:
            return 'rectangle'
    except Exception:
        return 'unknown'

def _draw_body_type_badge(frame, x, y, body_type):
    """Small pill badge showing body type label."""
    lbl = _BODY_TYPE_LABELS.get(body_type, body_type)
    col = _BODY_TYPE_COLORS.get(body_type, DIM)
    tw = _text_size(lbl, FONT, 0.34, 1)[0][0]
    bw = tw + 20
    _rrect(frame, x, y, x + bw, y + 20, 10, DARK)
    _rrect_border(frame, x, y, x + bw, y + 20, 10, col, 1)
    _rrect(frame, x, y, x + 4, y + 20, 2, col)
    _text(frame, lbl, (x + 10, y + 15), col, 0.34, 1)
    return bw


# ── Per-measurement confidence grade ─────────────────────────────────────────
def _confidence_grade(sigma):
    """Convert sigma uncertainty → letter grade A/B/C/D with colour."""
    if sigma is None or sigma < 0:
        return '', DIM
    if sigma < 0.8:
        return 'A', GREEN
    elif sigma < 1.6:
        return 'B', GOLD
    elif sigma < 2.8:
        return 'C', (45, 130, 215)   # orange
    else:
        return 'D', RED

def _draw_confidence_badge(frame, x, y, sigma):
    """Draw A/B/C/D letter badge at given position."""
    grade, col = _confidence_grade(sigma)
    if not grade:
        return
    r = 8
    cv2.circle(frame, (x + r, y + r), r, DARK, -1)
    cv2.circle(frame, (x + r, y + r), r, col, 1)
    gw = _text_size(grade, FONT, 0.30, 1)[0][0]
    _put_text(frame, grade, (x + r - gw // 2, y + r + 4), FONT, 0.30, col, 1)


# ── Non-blocking native folder picker ─────────────────────────────────────────
def launch_folder_picker(initial_dir):
    """Opens a native OS folder-chooser in a subprocess — does not block OpenCV."""
    script = (
        "import tkinter as tk; from tkinter import filedialog\n"
        "root=tk.Tk(); root.withdraw(); root.wm_attributes('-topmost',True)\n"
        f"f=filedialog.askdirectory(title='TailorCam — Choose save folder',"
        f"initialdir={repr(initial_dir)})\n"
        "root.destroy(); print(f or '',end='')\n"
    )
    return subprocess.Popen([sys.executable, '-c', script],
                            stdout=subprocess.PIPE, text=True)


def pick_photos():
    """
    Ask for a FRONT photo, then a SIDE photo (side may be cancelled).
    Blocks until the dialogs close. Returns a list of 0–2 file paths.
    """
    script = (
        "import tkinter as tk; from tkinter import filedialog\n"
        "root=tk.Tk(); root.withdraw(); root.wm_attributes('-topmost',True)\n"
        "ft=[('Photos','*.jpg *.jpeg *.png *.bmp *.webp'),('All files','*.*')]\n"
        "a=filedialog.askopenfilename(title='TailorCam - choose the FRONT photo (arms slightly away)',filetypes=ft)\n"
        "b=filedialog.askopenfilename(title='TailorCam - choose the SIDE photo (Cancel = front only)',filetypes=ft) if a else ''\n"
        "root.destroy(); print(a or ''); print(b or '')\n"
    )
    try:
        out = subprocess.run([sys.executable, '-c', script], capture_output=True,
                             text=True, encoding='utf-8', timeout=600).stdout
    except Exception as exc:
        print(f'[Upload] picker failed: {exc}')
        return []
    return [p.strip() for p in out.splitlines() if p.strip()]


def load_photo(path, max_side=1920):
    """Read a photo (any path, phone rotation applied) as a BGR image, or None."""
    try:
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        print(f'[Upload] cannot read {path}: {exc}')
        return None
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


# ── PDF generation ────────────────────────────────────────────────────────────
def save_pdf(measurements, customer_name='Customer', save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(False)

    # Header (deep charcoal + warm amber gold theme)
    pdf.set_fill_color(8, 12, 18)
    pdf.rect(0, 0, 210, 42, 'F')
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(240, 165, 0)   # warm amber gold
    pdf.set_xy(0, 9)
    pdf.cell(210, 10, 'TAILORCAM', align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(140, 155, 170)
    pdf.set_xy(0, 24)
    pdf.cell(210, 8, 'Professional Body Measurement Report', align='C')
    pdf.set_draw_color(240, 165, 0)
    pdf.set_line_width(0.7)
    pdf.line(0, 42, 210, 42)

    # Customer bar
    pdf.set_fill_color(12, 20, 30)
    pdf.rect(0, 42, 210, 14, 'F')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(140, 160, 180)
    now = datetime.datetime.now().strftime('%d %B %Y  %H:%M')
    pdf.set_xy(14, 45)
    pdf.cell(90, 8, f'Customer:  {customer_name}')
    pdf.set_xy(108, 45)
    pdf.cell(90, 8, f'Date:  {now}', align='R')

    # Body diagram
    _pdf_body(pdf, cx=50, cy=172, scale=1.0)

    # Measurement table (right column)
    tx, ty = 110, 62
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(0, 185, 230)   # cyan
    pdf.set_xy(tx, ty)
    pdf.cell(88, 9, 'Body Measurements')
    pdf.set_draw_color(0, 185, 230)
    pdf.set_line_width(0.35)
    pdf.line(tx, ty+10, tx+88, ty+10)

    row_h = 16
    row_fills = [(18, 32, 48), (12, 20, 30)]
    accent_rgb = {
        'shoulder_width': (240, 165, 0),
        'chest':          (240, 165, 0),
        'waist':          (40, 195, 50),
        'back_length':    (240, 165, 0),
        'shirt_length':   (240, 165, 0),
        'sleeve_length':  (240, 165, 0),
        'neck':           (240, 165, 0),
    }

    for i, (key, label) in enumerate(MEASUREMENT_ORDER):
        val = measurements.get(key, '—')
        ry = ty + 14 + i * row_h
        r, g, b = row_fills[i % 2]
        pdf.set_fill_color(r, g, b)
        pdf.rect(tx, ry, 88, row_h, 'F')
        # accent left bar
        ar, ag, ab = accent_rgb.get(key, (200, 200, 200))
        pdf.set_fill_color(ar, ag, ab)
        pdf.rect(tx, ry, 2.5, row_h, 'F')
        # label
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(140, 140, 165)
        pdf.set_xy(tx+6, ry+4)
        pdf.cell(52, 8, label)
        # value
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(235, 235, 245)
        pdf.set_xy(tx+60, ry+3)
        pdf.cell(22, 9, str(val), align='R')
        # unit
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(100, 100, 130)
        pdf.set_xy(tx+83, ry+5)
        pdf.cell(10, 8, 'cm')

    # Note
    note_y = ty + 14 + len(MEASUREMENT_ORDER) * row_h + 6
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(90, 90, 110)
    pdf.set_xy(tx, note_y)
    pdf.multi_cell(88, 5,
        'Tip: Wear fitted clothing for best accuracy. '
        'Scan improves further once the correction model is trained.')

    # Footer
    pdf.set_fill_color(8, 12, 18)
    pdf.rect(0, 272, 210, 25, 'F')
    pdf.set_draw_color(240, 165, 0)
    pdf.set_line_width(0.4)
    pdf.line(0, 272, 210, 272)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(110, 140, 165)
    pdf.set_xy(0, 279)
    pdf.cell(210, 8, 'Generated by TailorCam  |  AI-powered body measurement system', align='C')

    fname = (f"TailorCam_{customer_name.replace(' ','_')}_"
             f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    path = os.path.join(save_dir, fname)
    pdf.output(path)
    return path


def _pdf_body(pdf, cx, cy, scale):
    def x(u): return cx + u * scale
    def y(u): return cy + u * scale

    pdf.set_draw_color(55, 55, 80)
    pdf.set_line_width(0.45)
    pdf.ellipse(x(-9), y(-102), 18*scale, 22*scale, 'D')
    pdf.line(x(-3), y(-80), x(-3), y(-68))
    pdf.line(x(3),  y(-80), x(3),  y(-68))
    pdf.line(x(-3), y(-70), x(-28), y(-65))
    pdf.line(x(3),  y(-70), x(28),  y(-65))
    pdf.line(x(-28),y(-65), x(-30), y(-15))
    pdf.line(x(28), y(-65), x(30),  y(-15))
    pdf.line(x(-30),y(-15), x(-27), y(28))
    pdf.line(x(30), y(-15), x(27),  y(28))
    pdf.line(x(-28),y(-65), x(-20), y(5))
    pdf.line(x(28), y(-65), x(20),  y(5))
    pdf.line(x(-20),y(5),   x(-24), y(28))
    pdf.line(x(20), y(5),   x(24),  y(28))
    pdf.line(x(-24),y(28),  x(24),  y(28))
    pdf.line(x(-15),y(28),  x(-14), y(80))
    pdf.line(x(15), y(28),  x(14),  y(80))
    pdf.line(x(-24),y(28),  x(-15), y(28))
    pdf.line(x(24), y(28),  x(15),  y(28))

    # annotation lines (amber gold theme)
    pdf.set_draw_color(240, 165, 0)
    pdf.set_line_width(0.22)
    ann = x(36)
    for yu, lbl in [(-65,'Shoulder Width'),(-35,'Chest Circ.'),(5,'Waist Circ.')]:
        py = y(yu)
        pdf.line(x(28), py, ann, py)
        pdf.set_fill_color(240, 165, 0)
        pdf.ellipse(ann-0.7, py-0.7, 1.4, 1.4, 'F')
        pdf.set_font('Helvetica','',6.5)
        pdf.set_text_color(200, 140, 30)
        pdf.set_xy(ann+1.5, py-2.5)
        pdf.cell(30, 5, lbl)

    # sleeve bracket
    lx = x(-50)
    pdf.line(lx, y(-65), lx, y(28))
    pdf.line(lx, y(-65), x(-28), y(-65))
    pdf.line(lx, y(28),  x(-27), y(28))
    pdf.set_font('Helvetica','',6)
    pdf.set_text_color(200, 140, 30)
    pdf.set_xy(lx-16, y(-20)-2.5)
    pdf.cell(14, 5, 'Sleeve', align='R')

    # shirt/back length bracket
    llx = x(-38)
    pdf.line(llx, y(-70), llx, y(28))
    pdf.line(llx-1.5, y(-70), llx+1.5, y(-70))
    pdf.line(llx-1.5, y(5),   llx+1.5, y(5))
    pdf.line(llx-1.5, y(28),  llx+1.5, y(28))
    pdf.set_xy(llx-20, y(-35)-2)
    pdf.cell(18, 5, 'Shirt', align='R')
    pdf.set_xy(llx-20, y(16)-2)
    pdf.cell(18, 5, 'Back', align='R')

    # neck
    ny = y(-74)
    pdf.line(x(3), ny, ann, ny)
    pdf.set_fill_color(240, 165, 0)
    pdf.ellipse(ann-0.7, ny-0.7, 1.4, 1.4, 'F')
    pdf.set_font('Helvetica','',6.5)
    pdf.set_text_color(200, 140, 30)
    pdf.set_xy(ann+1.5, ny-2.5)
    pdf.cell(30, 5, 'Neck Circ.')


# ── Photo-capture guide overlay ───────────────────────────────────────────────
def _scan_quality_warnings(frame, lighting_ok=True, distance_ok=True):
    """Overlay amber warning chips for lighting / distance issues."""
    h, w = frame.shape[:2]
    wx, wy = 12, 50
    warnings = []
    if not lighting_ok:
        warnings.append('LOW LIGHT -- move to brighter area')
    if not distance_ok:
        warnings.append('ADJUST DISTANCE -- step 1.5-2m from camera')
    for warn in warnings:
        ww = _text_size(warn, FONT, 0.42, 1)[0][0] + 28
        _alpha(frame, wx, wy, wx + ww, wy + 26, 8, (0, 20, 40), 0.88)
        _rrect_border(frame, wx, wy, wx + ww, wy + 26, 8, RED, 2)
        # Warning triangle icon
        tri = np.array([[wx+10, wy+22],[wx+4, wy+22],[wx+7, wy+6]], np.int32)
        cv2.fillPoly(frame, [tri], RED)
        _put_text(frame, '!', (wx+6, wy+21), FONT, 0.28, DARK, 1)
        _text(frame, warn, (wx + 18, wy + 19), RED, 0.42, 1)
        wy += 32


def _shade(frame, x0, y0, x1, y1, a=0.55):
    """Darken a strip of the camera image so overlay text stays readable."""
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(frame.shape[1], x1), min(frame.shape[0], y1)
    if x1 > x0 and y1 > y0:
        roi = frame[y0:y1, x0:x1]
        frame[y0:y1, x0:x1] = (roi * (1 - a) + np.array(DARK) * a).astype(np.uint8)


def _corner_guide(frame, x0, y0, x1, y1, col, L=36, t=4):
    """Four corner brackets marking where the body should stand."""
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), col, t, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), col, t, cv2.LINE_AA)


def draw_photo_guide(frame, view_name, instruction, match_score,
                     countdown_val=None, view_idx=0, height_est=None,
                     total_views=2, lighting_ok=True, distance_ok=True,
                     guide_elapsed=0.0, guide_timeout=15.0):
    """
    Live camera guide: step indicator + pose-match meter (top), corner guide
    for the body, instruction card + auto-capture timer (bottom), countdown ring.
    """
    h, w = frame.shape[:2]
    good, ok = match_score >= 0.85, match_score >= 0.45
    accent = GREEN if good else (GOLD if ok else GREY)
    names = {'FRONT': 'Front', 'LEFT': 'Side', 'RIGHT': 'Right side', 'BACK': 'Back'}
    order = ['FRONT', 'LEFT', 'RIGHT', 'BACK'][:total_views]

    # ── top bar ───────────────────────────────────────────────────────────────
    _shade(frame, 0, 0, w, 64, 0.62)
    x = 24
    for i, v in enumerate(order):
        done, cur = i < view_idx, i == view_idx
        cv2.circle(frame, (x + 12, 32), 12, GREEN if done else (GOLD if cur else CARD_HI), -1, cv2.LINE_AA)
        if done:
            cv2.line(frame, (x + 6, 32), (x + 10, 37), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x + 10, 37), (x + 18, 27), (255, 255, 255), 2, cv2.LINE_AA)
        else:
            _text(frame, str(i + 1), (x + 8, 37), (255, 255, 255), TS_CAP, 2)
        _text(frame, names.get(v, v.title()), (x + 30, 38), WHITE if (done or cur) else GREY, TS_LABEL, 2)
        x += 30 + _text_size(names.get(v, v), FONT, TS_LABEL, 2)[0][0] + 16
        if i < len(order) - 1:
            cv2.line(frame, (x - 6, 32), (x + 26, 32), BORDER, 2, cv2.LINE_AA)
            x += 36
    # pose-match meter
    lbl = 'Pose match'
    pct = f'{int(match_score * 100)}%'
    mw = 180
    mx0 = w - 24 - mw
    _text(frame, lbl, (mx0, 28), GREY, TS_CAP, 1)
    _text(frame, pct, (w - 24 - _text_size(pct, FONT, TS_LABEL, 2)[0][0], 28), accent, TS_LABEL, 2)
    _rrect(frame, mx0, 38, mx0 + mw, 44, 3, CARD_HI)
    _rrect(frame, mx0, 38, mx0 + max(6, int(mw * min(1.0, match_score))), 44, 3, accent)

    # ── body guide ────────────────────────────────────────────────────────────
    gy0 = 84
    gh = (h - 112 - 14) - gy0                         # between the top bar and the instruction card
    gw = int(gh * 0.42)
    gx0 = (w - gw) // 2
    _corner_guide(frame, gx0, gy0, gx0 + gw, gy0 + gh, accent)

    # ── quality warnings (lighting / distance) ────────────────────────────────
    warn = [m for m, bad in (('More light needed', not lighting_ok), ('Step back so the whole body fits', not distance_ok)) if bad]
    if warn:
        msg = '  ·  '.join(warn)
        tw = _text_size(msg, FONT, TS_LABEL, 2)[0][0]
        _rrect(frame, (w - tw) // 2 - 30, 76, (w + tw) // 2 + 16, 106, 15, PANEL)
        cv2.circle(frame, ((w - tw) // 2 - 14, 91), 5, AMBER, -1, cv2.LINE_AA)
        _text(frame, msg, ((w - tw) // 2, 97), AMBER, TS_LABEL, 2)

    # ── bottom instruction card ───────────────────────────────────────────────
    parts = [p.strip() for p in str(instruction).split('--') if p.strip()]
    title = parts[0][:1].upper() + parts[0][1:] if parts else names.get(view_name, view_name)
    sub = ' · '.join(parts[1:])
    if countdown_val is None:
        status = 'Hold still' if good else ('Almost there — match the outline' if ok else 'Stand inside the frame, whole body visible')
    else:
        status = 'Hold still — taking the photo'
    cw_ = min(720, w - 48)
    cx0, cy0 = (w - cw_) // 2, h - 112
    _rrect(frame, cx0, cy0, cx0 + cw_, cy0 + 88, 14, PANEL)
    _rrect_border(frame, cx0, cy0, cx0 + cw_, cy0 + 88, 14, BORDER, 1)
    cv2.circle(frame, (cx0 + 26, cy0 + 30), 6, accent, -1, cv2.LINE_AA)
    _text(frame, status, (cx0 + 42, cy0 + 36), accent if good else WHITE, TS_TITLE, 3)
    _text(frame, f'{title}' + (f'  ·  {sub}' if sub else ''), (cx0 + 42, cy0 + 64), GREY, TS_BODY, 1)
    if countdown_val is None and guide_elapsed > 5.0 and guide_timeout > 0:
        left = max(0.0, 1.0 - guide_elapsed / guide_timeout)
        _rrect(frame, cx0 + 16, cy0 + 78, cx0 + cw_ - 16, cy0 + 82, 2, CARD_HI)
        _rrect(frame, cx0 + 16, cy0 + 78, cx0 + 16 + max(4, int((cw_ - 32) * left)), cy0 + 82, 2, GOLD)
        secs = max(0, int(guide_timeout - guide_elapsed))
        t_ = f'auto-capture in {secs}s' if secs else 'capturing…'
        _text(frame, t_, (cx0 + cw_ - 20 - _text_size(t_, FONT, TS_CAP, 1)[0][0], cy0 + 36), DIM, TS_CAP, 1)

    # ── height + shortcuts (bottom corners) ───────────────────────────────────
    h_txt = f'Height {height_est:.0f} cm' if height_est else 'Height not set'
    hx = 24
    hw = _text_size(h_txt, FONT, TS_CAP, 2)[0][0]
    _rrect(frame, hx, h - 40, hx + hw + 20, h - 16, 12, PANEL)
    _text(frame, h_txt, (hx + 10, h - 23), WHITE if height_est else AMBER, TS_CAP, 2)
    hint_items = [('T', 'type height')] + ([('O', 'measure from photos')] if view_idx == 0 else [])
    hx2 = w - 24 - sum(_text_size(l, FONT, TS_CAP, 1)[0][0] + 44 for _, l in hint_items)
    _rrect(frame, hx2 - 12, h - 46, w - 14, h - 10, 12, PANEL)
    _hint_row(frame, hx2, h - 22, hint_items)

    # ── countdown ─────────────────────────────────────────────────────────────
    if countdown_val is not None:
        cx, cy = w // 2, h // 2 - 20
        ov = frame.copy()
        cv2.circle(ov, (cx, cy), 70, DARK, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        frac = (time.time() % 1.0)
        cv2.circle(frame, (cx, cy), 70, CARD_HI, 6, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (70, 70), -90, 0, int(360 * (1 - frac)), GREEN, 6, cv2.LINE_AA)
        cs = str(countdown_val)
        tw, th = _text_size(cs, FONT, 2.6, 3)[0]
        _text(frame, cs, (cx - tw // 2, cy + th // 2), (255, 255, 255), 2.6, 3)
    return frame


# ── Customer records (JSON) ────────────────────────────────────────────────────
def save_record(measurements, customer_name, save_dir, pdf_fname=''):
    """Append one measurement record to records.json in save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    records = load_records(save_dir)
    records.append({
        'name': customer_name,
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'pdf':  pdf_fname,
        **{k: measurements.get(k) for k in (
            'shoulder_width', 'chest', 'waist',
            'back_length', 'shirt_length', 'sleeve_length', 'neck')},
    })
    path = os.path.join(save_dir, 'records.json')
    with open(path, 'w') as f:
        json.dump(records, f, indent=2)


def load_records(save_dir):
    path = os.path.join(save_dir, 'records.json')
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ── Home screen ───────────────────────────────────────────────────────────────
def _kbd(frame, x, y, key):
    """Small keyboard-key chip (shortcut hint). Returns its width."""
    kw = _text_size(key, FONT, TS_CAP, 1)[0][0] + 12
    _rrect(frame, x, y, x + kw, y + 20, 5, CARD_HI)
    _text(frame, key, (x + 6, y + 15), GREY, TS_CAP, 1)
    _register((x - 4, y - 6, x + kw + 4, y + 26), key)
    return kw


def _button(frame, x1, y1, x2, y2, label, key=None, primary=False, hover=False, danger=False):
    """Standard button: primary = filled accent, otherwise quiet card with border."""
    if primary:
        _rrect(frame, x1, y1, x2, y2, 10, GOLD_HI if hover else GOLD)
        col = (255, 255, 255)
    else:
        _rrect(frame, x1, y1, x2, y2, 10, CARD_HI if hover else CARD)
        _rrect_border(frame, x1, y1, x2, y2, 10, (RED if danger else GOLD) if hover else BORDER, 1)
        col = RED if (danger and hover) else WHITE
    if key:
        _register((x1, y1, x2, y2), key)
    lw = _text_size(label, FONT, TS_BODY, 2)[0][0]
    kw = (_text_size(key, FONT, TS_CAP, 1)[0][0] + 20) if key else 0
    cx = (x1 + x2 - lw - kw) // 2
    _text(frame, label, (cx, (y1 + y2) // 2 + 6), col, TS_BODY, 2)
    if key:
        kx = cx + lw + 10
        _rrect(frame, kx, (y1 + y2) // 2 - 10, kx + kw - 10, (y1 + y2) // 2 + 10, 5,
               tuple(min(255, c + 40) for c in GOLD) if primary else CARD_HI)
        _text(frame, key, (kx + 6, (y1 + y2) // 2 + 5), (255, 255, 255) if primary else GREY, TS_CAP, 1)


def draw_home_screen(frame, records=None, t=0.0, mouse_xy=None, stats=None):
    """
    Dashboard: header · greeting · 4 stat cards · actions · recent customers +
    order pipeline. Returns (frame, btns) with the same button keys as before.
    """
    h, w = frame.shape[:2]
    btns = {}
    recs = list(reversed(records or []))
    n = len(recs)
    mx, my = mouse_xy if mouse_xy else (-1, -1)
    hov = lambda x1, y1, x2, y2: x1 <= mx <= x2 and y1 <= my <= y2
    P = 24                                            # page margin

    frame[:] = DARK

    # ── Header ────────────────────────────────────────────────────────────────
    HDR = 64
    _rrect(frame, 0, 0, w, HDR, 0, PANEL)
    cv2.line(frame, (0, HDR), (w, HDR), BORDER, 1)
    _rrect(frame, P, 16, P + 32, 48, 8, GOLD)                       # logo mark
    _text(frame, 'T', (P + 10, 40), (255, 255, 255), 0.62, 3)
    _text(frame, 'TailorCam', (P + 44, 38), WHITE, 0.62, 3)
    _text(frame, 'Body measurement studio', (P + 44 + _text_size('TailorCam', FONT, 0.62, 3)[0][0] + 12, 37),
          GREY, TS_CAP, 1)
    ex = (w - P - 84, 14, w - P, 50)
    se = (ex[0] - 12 - 104, 14, ex[0] - 12, 50)
    _button(frame, *se, 'Settings', 'E', hover=hov(*se))
    _button(frame, *ex, 'Exit', 'Q', hover=hov(*ex), danger=True)
    btns['settings'], btns['exit'] = se, ex
    now = datetime.datetime.now()
    clock = now.strftime('%a %d %b  ·  %H:%M')
    _text(frame, clock, (se[0] - 20 - _text_size(clock, FONT, TS_CAP, 1)[0][0], 37), GREY, TS_CAP, 1)

    # ── Greeting ──────────────────────────────────────────────────────────────
    hr = now.hour
    greet = 'Good morning' if hr < 12 else ('Good afternoon' if hr < 17 else 'Good evening')
    y = HDR + 44
    _text(frame, greet, (P, y), WHITE, TS_H1, 3)
    _text(frame, 'Here is your shop today', (P, y + 24), GREY, TS_BODY, 1)

    # ── Stat cards ────────────────────────────────────────────────────────────
    chests = [r.get('chest') for r in recs if r.get('chest')]
    today = datetime.date.today().isoformat()
    scans_today = sum(1 for r in recs if str(r.get('date', '')).startswith(today))
    st = stats or {}
    cards = [('Customers', str(st.get('total_customers', n)), 'in your records'),
             ('Scans today', str(scans_today), today),
             ('Average chest', f'{sum(chests) / len(chests):.0f} cm' if chests else '—', f'over {len(chests)} scans'),
             ('Active orders', str(st.get('active_orders', 0)),
              f"revenue {int(st['revenue']):,} BDT" if st.get('revenue') else 'no revenue yet')]
    CY, CH, gap = y + 48, 96, 16
    cw = (w - 2 * P - gap * (len(cards) - 1)) // len(cards)
    for i, (lbl, val, sub) in enumerate(cards):
        x = P + i * (cw + gap)
        _rrect(frame, x, CY, x + cw, CY + CH, 12, CARD)
        _rrect_border(frame, x, CY, x + cw, CY + CH, 12, BORDER, 1)
        _text(frame, lbl, (x + 18, CY + 28), GREY, TS_LABEL, 1)
        _text(frame, val, (x + 18, CY + 66), WHITE, 1.0, 3)
        _text(frame, sub, (x + 18, CY + 86), DIM, TS_CAP, 1)

    # ── Actions ───────────────────────────────────────────────────────────────
    AY, AH = CY + CH + 20, 52
    acts = [('New scan', 'N', 'new_cta', True), ('Upload photos', 'O', 'upload', False),
            ('Customers', 'C', 'custs', False), ('History', 'H', 'history', False), ('Reports', 'P', 'reports', False)]
    aw = (w - 2 * P - gap * (len(acts) - 1)) // len(acts)
    for i, (lbl, key, bid, prim) in enumerate(acts):
        x = P + i * (aw + gap)
        r_ = (x, AY, x + aw, AY + AH)
        _button(frame, *r_, lbl, key, primary=prim, hover=hov(*r_))
        btns[bid] = r_
    btns['new'] = btns['new_cta']

    # ── Recent customers ──────────────────────────────────────────────────────
    TY = AY + AH + 32
    LW = int(w * 0.62)
    _text(frame, 'Recent customers', (P, TY), WHITE, TS_TITLE, 2)
    tag = f'{n} records' if n else 'no records yet'
    _text(frame, tag, (LW - _text_size(tag, FONT, TS_CAP, 1)[0][0], TY), DIM, TS_CAP, 1)
    top = TY + 14
    _rrect(frame, P, top, LW, h - P - 30, 12, CARD)
    _rrect_border(frame, P, top, LW, h - P - 30, 12, BORDER, 1)
    cols = [('Customer', P + 20), ('Date', P + int(LW * 0.38)), ('Chest', P + int(LW * 0.56)),
            ('Waist', P + int(LW * 0.70)), ('Size', P + int(LW * 0.84))]
    for lbl, cx in cols:
        _text(frame, lbl.upper(), (cx, top + 26), DIM, TS_CAP, 1)
    cv2.line(frame, (P + 12, top + 38), (LW - 12, top + 38), BORDER, 1)
    row_h, ry0 = 44, top + 40
    rows_fit = max(0, (h - P - 30 - ry0 - 6) // row_h)
    if not recs:
        msg = 'No scans yet — press N to start your first scan'
        _text(frame, msg, ((P + LW - _text_size(msg, FONT, TS_BODY, 1)[0][0]) // 2, ry0 + 60), GREY, TS_BODY, 1)
    for i, rec in enumerate(recs[:rows_fit]):
        ry = ry0 + i * row_h
        if hov(P, ry, LW, ry + row_h):
            _rrect(frame, P + 6, ry + 2, LW - 6, ry + row_h - 2, 8, CARD_HI)
        mid = ry + row_h // 2 + 6
        name = rec.get('name') or rec.get('customer_name') or 'Customer'
        _avatar_circle(frame, cols[0][1] + 13, ry + row_h // 2, 13, name)
        _text(frame, name[:24], (cols[0][1] + 34, mid), WHITE, TS_BODY, 1)
        _text(frame, str(rec.get('date', ''))[:10], (cols[1][1], mid), GREY, TS_LABEL, 1)
        for (key_r, cx) in (('chest', cols[2][1]), ('waist', cols[3][1])):
            v = rec.get(key_r)
            if v is not None:
                s_ = f'{float(v):.0f}'
                _text(frame, s_, (cx, mid), WHITE, TS_BODY, 2)
                _text(frame, 'cm', (cx + _text_size(s_, FONT, TS_BODY, 2)[0][0] + 4, mid), DIM, TS_CAP, 1)
        sz = _size_label('chest', rec.get('chest'))
        if sz:
            sw = _text_size(sz, FONT, TS_CAP, 2)[0][0]
            _rrect(frame, cols[4][1], ry + 12, cols[4][1] + sw + 16, ry + row_h - 12, 6, CARD_HI)
            _text(frame, sz, (cols[4][1] + 8, mid - 1), WHITE, TS_CAP, 2)
        if i < rows_fit - 1:
            cv2.line(frame, (P + 16, ry + row_h), (LW - 16, ry + row_h), (BORDER[0] - 8, BORDER[1] - 8, BORDER[2] - 8), 1)

    # ── Order pipeline ────────────────────────────────────────────────────────
    RX = LW + 24
    RW = w - P - RX
    _text(frame, 'Order pipeline', (RX, TY), WHITE, TS_TITLE, 2)
    stages = [('Received', 'stage_received'), ('Cutting', 'stage_cutting'), ('Stitching', 'stage_stitching'),
              ('Finishing', 'stage_finishing'), ('Ready', 'stage_ready'), ('Delivered', 'stage_delivered')]
    _rrect(frame, RX, top, RX + RW, h - P - 30, 12, CARD)
    _rrect_border(frame, RX, top, RX + RW, h - P - 30, 12, BORDER, 1)
    sh = min(44, (h - P - 30 - top - 12) // len(stages))
    for i, (lbl, key) in enumerate(stages):
        sy = top + 8 + i * sh
        cnt = int(st.get(key, 0) or 0)
        col = GREEN if (lbl == 'Ready' and cnt) else (GOLD if cnt else BORDER)
        cv2.circle(frame, (RX + 22, sy + sh // 2), 5, col, -1, cv2.LINE_AA)
        _text(frame, lbl, (RX + 38, sy + sh // 2 + 6), WHITE if cnt else GREY, TS_BODY, 1)
        cs = str(cnt) if cnt else '—'
        _text(frame, cs, (RX + RW - 20 - _text_size(cs, FONT, TS_BODY, 2)[0][0], sy + sh // 2 + 6),
              WHITE if cnt else DIM, TS_BODY, 2)
        if i < len(stages) - 1:
            cv2.line(frame, (RX + 16, sy + sh), (RX + RW - 16, sy + sh), (BORDER[0] - 8, BORDER[1] - 8, BORDER[2] - 8), 1)

    # ── Shortcut legend ───────────────────────────────────────────────────────
    lx, ly = P, h - P - 18
    for key, lbl in (('N', 'New scan'), ('O', 'Upload photos'), ('C', 'Customers'), ('V', 'Inventory'), ('H', 'History'), ('P', 'Reports'),
                     ('E', 'Settings'), ('Q', 'Exit')):
        lx += _kbd(frame, lx, ly - 4, key) + 6
        _text(frame, lbl, (lx, ly + 11), DIM, TS_CAP, 1)
        lx += _text_size(lbl, FONT, TS_CAP, 1)[0][0] + 18
    return frame, btns


# ── History / past measurements screen ────────────────────────────────────────
# ── Customer selection screen ─────────────────────────────────────────────────
def draw_customer_select_screen(frame, customers, query='', scroll=0, mouse_xy=None,
                                loyalty_data=None, photo_cache=None):
    """
    Search + select customer before a scan. Returns (frame, btns).
    loyalty_data: optional dict {customer_id: {'points': int, 'tier': str}}
    photo_cache:  optional dict {customer_id: BGR numpy array} for profile avatars
    """
    h, w = frame.shape[:2]
    btns = {}
    mx, my = mouse_xy if mouse_xy else (-1, -1)

    frame[:] = DARK

    # Header
    _rrect(frame, 0, 0, w, 56, 0, PANEL)
    cv2.line(frame, (0, 56), (w, 56), BORDER, 1)
    lx_h = 24
    pts_h = np.array([[lx_h,8],[lx_h+8,28],[lx_h,48],[lx_h-8,28]], np.int32)
    cv2.fillPoly(frame, [pts_h], GOLD)
    _brand(frame, 30)
    cv2.line(frame, (160, 14), (160, 42), BORDER, 1)
    _text(frame, 'SELECT CUSTOMER', (174, 36), WHITE, 0.56, 1, FONT_DUP)

    # Back button
    bk_x = w - 258
    _rrect(frame, bk_x, 11, bk_x + 100, 45, 8, CARD)
    _rrect_border(frame, bk_x, 11, bk_x + 100, 45, 8, GOLD, 1)
    _text(frame, '<  BACK', (bk_x + 14, 34), GOLD, 0.48, 1)
    btns['back'] = (bk_x, 11, bk_x + 100, 45)

    # New customer button
    nc_x = w - 148
    _rrect(frame, nc_x, 11, nc_x + 130, 45, 8, (11, 22, 11))
    _rrect_border(frame, nc_x, 11, nc_x + 130, 45, 8, GREEN, 1)
    _text(frame, '+  NEW  N', (nc_x + 12, 34), GREEN, 0.44, 1)
    btns['new_customer'] = (nc_x, 11, nc_x + 130, 45)
    btns.update(_page_header(frame, 'Customers', [('new_customer', 'New customer', 'N', 'primary'),
                                                  ('back', 'Back', 'Esc', 'normal')], mouse_xy, height=56))

    # Search bar
    sb_x, sb_y, sb_w, sb_h = 14, 66, w - 28, 36
    _rrect(frame, sb_x, sb_y, sb_x + sb_w, sb_y + sb_h, 8, CARD)
    _rrect_border(frame, sb_x, sb_y, sb_x + sb_w, sb_y + sb_h, 8, GOLD, 1)
    _text(frame, 'Search:', (sb_x + 10, sb_y + sb_h - 10), GREY, 0.36, 1)
    q_disp = (query + '|') if query else 'Type name or phone...'
    _text(frame, q_disp, (sb_x + 72, sb_y + sb_h - 10),
          WHITE if query else DIM, 0.44, 1)

    # Customer cards
    card_x  = 14
    card_y  = sb_y + sb_h + 10
    card_w  = w - 28
    card_h  = 56
    card_gap = 6
    vis     = max(1, (h - card_y - 20) // (card_h + card_gap))
    shown   = customers[scroll: scroll + vis]

    if not shown:
        mid_e = (card_y + h) // 2
        _centered(frame, 'No customers found', mid_e - 16, GREY, 0.58, 1)
        _centered(frame, 'Press  N  to add a new customer', mid_e + 18, DIM, 0.42, 1)
    else:
        for i, cust in enumerate(shown):
            cy_c = card_y + i * (card_h + card_gap)
            hov = (card_x <= mx <= card_x + card_w and cy_c <= my <= cy_c + card_h)
            _rrect(frame, card_x, cy_c, card_x + card_w, cy_c + card_h, 10, CARD)
            _rrect_border(frame, card_x, cy_c, card_x + card_w, cy_c + card_h, 10,
                          GOLD if hov else BORDER, 2 if hov else 1)

            name_c  = cust.get('name', 'Unknown')
            cust_id_av = cust.get('id')
            photo_img = photo_cache.get(cust_id_av) if photo_cache and cust_id_av else None
            av_cx = card_x + 28
            av_cy = cy_c + card_h // 2
            av_r  = 18
            if photo_img is not None:
                try:
                    av_d = av_r * 2
                    ph, pw = photo_img.shape[:2]
                    sc = av_d / max(ph, pw)
                    ph2, pw2 = max(1, int(ph * sc)), max(1, int(pw * sc))
                    thumb = cv2.resize(photo_img, (pw2, ph2))
                    mask = np.zeros((av_d, av_d), np.uint8)
                    cv2.circle(mask, (av_r, av_r), av_r, 255, -1)
                    ry0, ry1 = max(0, av_cy - av_r), min(frame.shape[0], av_cy + av_r)
                    rx0, rx1 = max(0, av_cx - av_r), min(frame.shape[1], av_cx + av_r)
                    d_h, d_w = ry1 - ry0, rx1 - rx0
                    if d_h > 0 and d_w > 0:
                        th2 = cv2.resize(thumb, (d_w, d_h))
                        m2  = cv2.resize(mask,  (d_w, d_h))
                        roi = frame[ry0:ry1, rx0:rx1]
                        m3  = m2[:, :, None] / 255.0
                        frame[ry0:ry1, rx0:rx1] = (th2 * m3 + roi * (1 - m3)).astype(np.uint8)
                        cv2.circle(frame, (av_cx, av_cy), av_r, GOLD, 1)
                except Exception:
                    _avatar_circle(frame, av_cx, av_cy, av_r, name_c)
            else:
                _avatar_circle(frame, av_cx, av_cy, av_r, name_c)
            mid_c = cy_c + card_h // 2
            _text(frame, name_c[:30], (card_x + 56, mid_c + 2), WHITE, 0.52, 1)
            phone_c = cust.get('phone', '')
            if phone_c:
                _text(frame, phone_c, (card_x + 56, mid_c + 19), GREY, 0.38, 1)
            date_c = str(cust.get('updated_at', ''))[:10]
            # Loyalty badge (right side, before arrow)
            cust_id_c = cust.get('id')
            loy_pts = 0
            if loyalty_data and cust_id_c and cust_id_c in loyalty_data:
                loy_pts = loyalty_data[cust_id_c].get('points', 0)
            if loy_pts > 0:
                draw_loyalty_badge(frame, card_x + card_w - 120, mid_c - 9, loy_pts)
            if date_c:
                dtw_c = _text_size(date_c, FONT, 0.34, 1)[0][0]
                _text(frame, date_c,
                      (card_x + card_w - dtw_c - 130 if loy_pts > 0 else card_x + card_w - dtw_c - 28,
                       mid_c + 2), DIM, 0.34, 1)
            cv2.arrowedLine(frame,
                            (card_x + card_w - 28, cy_c + card_h // 2),
                            (card_x + card_w - 14, cy_c + card_h // 2),
                            GOLD if hov else BORDER, 2, tipLength=0.55)
            btns[f'cust_{scroll + i}'] = (card_x, cy_c, card_x + card_w, cy_c + card_h)

    # Scroll bar
    total_c = len(customers)
    if total_c > vis:
        sb_bar_h = h - card_y - 10
        sb_bar_y = card_y
        _rrect(frame, w - 8, sb_bar_y, w - 4, sb_bar_y + sb_bar_h, 3, BORDER)
        fill_frac = vis / max(total_c, 1)
        fill_top  = int(sb_bar_h * scroll / max(total_c - vis, 1))
        fill_sz   = max(20, int(sb_bar_h * fill_frac))
        _rrect(frame, w - 8, sb_bar_y + fill_top,
               w - 4, sb_bar_y + fill_top + fill_sz, 3, GOLD)

    return frame, btns


# ── Weight + age input overlay ────────────────────────────────────────────────
# ── Order creation screen ─────────────────────────────────────────────────────
def draw_order_screen(frame, customer_name='', measurements=None,
                      garment_type='t_shirt', style_notes='', fabric='',
                      colour='', price='', delivery='', active_field='garment',
                      assigned_to='', staff_list=None):
    """
    Create a new order with measurements summary on the left.
    assigned_to: name of the assigned tailor/staff
    staff_list: list of staff name strings for the display hint
    """
    h, w = frame.shape[:2]
    frame[:] = DARK

    # Header
    _rrect(frame, 0, 0, w, 56, 0, PANEL)
    cv2.line(frame, (0, 56), (w, 56), BORDER, 1)
    lx_o = 24
    pts_o = np.array([[lx_o,8],[lx_o+8,28],[lx_o,48],[lx_o-8,28]], np.int32)
    cv2.fillPoly(frame, [pts_o], GOLD)
    _brand(frame, 30)
    cv2.line(frame, (160, 14), (160, 42), BORDER, 1)
    _text(frame, 'CREATE ORDER', (174, 36), WHITE, 0.56, 1, FONT_DUP)
    if customer_name:
        _text(frame, f'for  {customer_name[:24]}', (174, 50), GREY, 0.32, 1)

    # Back + Confirm buttons
    bk_xo = w - 268
    _rrect(frame, bk_xo, 11, bk_xo + 100, 45, 8, CARD)
    _rrect_border(frame, bk_xo, 11, bk_xo + 100, 45, 8, GOLD, 1)
    _text(frame, '<  BACK', (bk_xo + 14, 34), GOLD, 0.48, 1)

    cf_xo = w - 158
    _rrect(frame, cf_xo, 11, cf_xo + 148, 45, 8, (11, 22, 11))
    _rrect_border(frame, cf_xo, 11, cf_xo + 148, 45, 8, GREEN, 2)
    _text(frame, 'CONFIRM  Enter', (cf_xo + 12, 34), GREEN, 0.44, 1)
    _page_header(frame, 'New order', [('back', 'Back', 'Esc', 'normal'), ('ok', 'Confirm order', 'Enter', 'primary')],
                 height=56)

    # Left: measurements summary panel
    mp_x, mp_y = 14, 66
    mp_w = 290
    _rrect(frame, mp_x, mp_y, mp_x + mp_w, h - 20, 10, CARD)
    _rrect_border(frame, mp_x, mp_y, mp_x + mp_w, h - 20, 10, BORDER)
    _text(frame, 'Measurements', (mp_x + 10, mp_y + 22), WHITE, TS_LABEL, 2)
    cv2.line(frame, (mp_x + 10, mp_y + 28), (mp_x + mp_w - 10, mp_y + 28), BORDER, 1)

    row_m = 26
    for i, (key_m, lbl_m) in enumerate(MEASUREMENT_ORDER):
        ry_m = mp_y + 36 + i * row_m
        if ry_m + row_m > h - 24:
            break
        a_m = _ACCENTS.get(key_m, WHITE)
        _text(frame, lbl_m, (mp_x + 14, ry_m + 14), GREY, 0.36, 1)
        v_m = (measurements or {}).get(key_m)
        if v_m is not None:
            vs = str(v_m)
            vw_m = _text_size(vs, FONT_DUP, 0.54, 1)[0][0]
            _put_text(frame, vs, (mp_x + mp_w - vw_m - 22, ry_m + 16),
                        FONT_DUP, 0.54, WHITE, 1)
            _text(frame, 'cm', (mp_x + mp_w - 16, ry_m + 13), DIM, 0.26, 1)

    # Right: order form fields
    fx_o = mp_x + mp_w + 14
    fw_o = w - fx_o - 14

    _GARMENT_NAMES = {'t_shirt':'T-Shirt','shirt':'Shirt','sweater':'Sweater',
                      'jacket':'Jacket','tight_fit':'Tight Fit'}
    g_disp_o = _GARMENT_NAMES.get(garment_type, garment_type)

    # Compute delivery countdown if date is set
    delivery_days = None
    delivery_overdue = False
    if delivery:
        try:
            d_date = datetime.date.fromisoformat(delivery[:10])
            delivery_days = (d_date - datetime.date.today()).days
            delivery_overdue = delivery_days < 0
        except Exception:
            pass

    # Staff hint (show available names)
    staff_hint = 'e.g. Main Tailor'
    if staff_list:
        staff_hint = ' / '.join(staff_list[:3])

    field_defs = [
        ('Garment Type',  g_disp_o,    'garment',     'e.g. T-Shirt'),
        ('Style Notes',   style_notes, 'notes',       'e.g. Slim fit, V-neck'),
        ('Fabric',        fabric,      'fabric',      'e.g. Cotton'),
        ('Colour',        colour,      'colour',      'e.g. Navy blue'),
        ('Price (BDT)',   price,       'price',       'e.g. 1500'),
        ('Delivery Date', delivery,    'delivery',    'YYYY-MM-DD'),
        ('Assign To',     assigned_to, 'assigned_to', staff_hint),
    ]
    field_h = 54
    for i, (lbl_f, val_f, fid, hint_f) in enumerate(field_defs):
        fy_f = 66 + i * (field_h + 6)
        if fy_f + field_h > h - 40:
            break
        act_f = (active_field == fid)
        _rrect(frame, fx_o, fy_f, fx_o + fw_o, fy_f + field_h, 8, DARK)
        _rrect_border(frame, fx_o, fy_f, fx_o + fw_o, fy_f + field_h, 8,
                      GOLD if act_f else BORDER, 2 if act_f else 1)
        _text(frame, lbl_f, (fx_o + 10, fy_f + 16), GREY, 0.34, 1)
        disp_f = (val_f + '|') if act_f else val_f
        _text(frame, disp_f[:40] if disp_f else hint_f,
              (fx_o + 10, fy_f + 42), WHITE if val_f else DIM, 0.50, 1)

        # Delivery countdown / overdue badge
        if fid == 'delivery' and delivery_days is not None:
            if delivery_overdue:
                badge = f'OVERDUE {abs(delivery_days)}d'
                badge_col = RED
            elif delivery_days == 0:
                badge = 'DUE TODAY'
                badge_col = (45, 130, 215)
            else:
                badge = f'{delivery_days}d left'
                badge_col = GREEN if delivery_days > 3 else (45, 130, 215)
            bw_d = _text_size(badge, FONT, 0.34, 1)[0][0]
            bx_d = fx_o + fw_o - bw_d - 22
            _rrect(frame, bx_d - 4, fy_f + 4, bx_d + bw_d + 10, fy_f + 22, 5, DARK)
            _rrect_border(frame, bx_d - 4, fy_f + 4, bx_d + bw_d + 10, fy_f + 22, 5, badge_col, 1)
            _text(frame, badge, (bx_d, fy_f + 17), badge_col, 0.34, 1)

    _text(frame, 'Tab = next field   Enter = confirm   Esc = cancel',
          (fx_o, h - 14), DIM, 0.34, 1)

    return frame


# ── Reports / analytics dashboard ────────────────────────────────────────────
# ── Settings screen ───────────────────────────────────────────────────────────
def draw_settings_screen(frame, shop_name='TailorCam Shop', currency='BDT',
                         use_inches=False, camera_idx=0, mouse_xy=None):
    """
    Settings / configuration screen.
    Returns (frame, btns).
    shop_name: editable shop name
    currency: 'BDT' | 'USD' | 'INR' | 'EUR'
    use_inches: bool for cm/in toggle
    camera_idx: int camera index
    """
    h, w = frame.shape[:2]
    btns = {}
    mx, my = mouse_xy if mouse_xy else (-1, -1)

    frame[:] = DARK

    # Header
    _rrect(frame, 0, 0, w, 56, 0, PANEL)
    cv2.line(frame, (0, 56), (w, 56), BORDER, 1)
    lx_s = 24
    pts_s = np.array([[lx_s,8],[lx_s+8,28],[lx_s,48],[lx_s-8,28]], np.int32)
    cv2.fillPoly(frame, [pts_s], GOLD)
    _brand(frame, 30)
    cv2.line(frame, (160, 14), (160, 42), BORDER, 1)
    _text(frame, 'SETTINGS', (174, 36), WHITE, 0.56, 1, FONT_DUP)

    # Back button
    bk_xs = w - 136
    _rrect(frame, bk_xs, 11, bk_xs + 118, 45, 8, CARD)
    _rrect_border(frame, bk_xs, 11, bk_xs + 118, 45, 8, GOLD, 1)
    _text(frame, '<  HOME  H', (bk_xs + 12, 34), GOLD, 0.44, 1)
    btns['back'] = (bk_xs, 11, bk_xs + 118, 45)
    btns.update(_page_header(frame, 'Settings', [('back', 'Home', 'H', 'normal')], mouse_xy, height=56))

    # Settings content area
    col_w = (w - 28) // 2
    left_x, right_x = 14, 14 + col_w + 6

    # ── Section helper ────────────────────────────────────────────────────────
    def section(x, y, title, col_w_s=col_w):
        _text(frame, title.title(), (x + 2, y + 17), GREY, TS_LABEL, 2)

    def setting_row(x, y, label, value, row_w=col_w, accent_col=WHITE, active=False):
        """Draw one setting row. Returns (x1,y1,x2,y2) for click."""
        row_h = 44
        bg = CARD if active else DARK
        _rrect(frame, x, y, x + row_w, y + row_h, 8, CARD_HI if active else CARD)
        _rrect_border(frame, x, y, x + row_w, y + row_h, 8, GOLD if active else BORDER, 2 if active else 1)
        _text(frame, label, (x + 12, y + 18), GREY, TS_CAP, 1)
        _text(frame, value, (x + 12, y + 37), RED if accent_col == RED else WHITE, TS_BODY, 2)
        # Arrow on right
        cv2.arrowedLine(frame, (x + row_w - 24, y + row_h // 2),
                        (x + row_w - 10, y + row_h // 2),
                        GOLD if active else BORDER, 2, tipLength=0.55)
        return (x, y, x + row_w, y + row_h)

    # ── Left column: Shop & Display ───────────────────────────────────────────
    sy = 68
    section(left_x, sy, 'SHOP PROFILE')
    sy += 26
    btns['shop_name'] = setting_row(left_x, sy, 'Shop Name', shop_name[:24] + ('|' if False else ''), accent_col=WHITE)
    sy += 50
    btns['currency'] = setting_row(left_x, sy, 'Currency', currency, accent_col=GOLD)
    sy += 50

    section(left_x, sy, 'DISPLAY')
    sy += 26
    unit_val = 'Inches (in)' if use_inches else 'Centimetres (cm)'
    unit_col = TEAL if use_inches else GREEN
    btns['units'] = setting_row(left_x, sy, 'Measurement Units', unit_val, accent_col=unit_col)
    sy += 50

    cam_val = f'Camera {camera_idx}'
    btns['camera'] = setting_row(left_x, sy, 'Camera Input', cam_val, accent_col=TEAL)
    sy += 50

    # ── Right column: Workflow ────────────────────────────────────────────────
    sy2 = 68
    section(right_x, sy2, 'WORKFLOW')
    sy2 += 26

    wf_rows = [
        ('Auto-Advance Results', '15 seconds'),
        ('Default Garment Type', 'T-Shirt'),
        ('WhatsApp Number',      'Not set'),
        ('Cloud Backup',         'Disabled'),
    ]
    for lbl_w, val_w in wf_rows:
        btns[f'wf_{lbl_w[:8].lower().replace(" ","_")}'] = setting_row(
            right_x, sy2, lbl_w, val_w, row_w=col_w, accent_col=GREY)
        sy2 += 50

    section(right_x, sy2, 'DATA')
    sy2 += 26

    data_rows = [
        ('Export All Records', 'CSV / JSON'),
        ('Clear All Data',     'Danger zone'),
    ]
    for lbl_d, val_d in data_rows:
        col_d = RED if 'Clear' in lbl_d else TEAL
        btns[f'data_{lbl_d[:5].lower().replace(" ","_")}'] = setting_row(
            right_x, sy2, lbl_d, val_d, row_w=col_w, accent_col=col_d)
        sy2 += 50

    # Version footer
    _centered(frame, 'TailorCam v2.0  --  AI Professional Body Measurement System',
              h - 20, DIM, 0.34, 1)

    return frame, btns


# ── Print order label (PDF A6) ────────────────────────────────────────────────
def print_order_label(customer_name, garment_type, delivery_date='',
                      style_notes='', price='', measurement_id=None,
                      save_dir='.'):
    """
    Generate a small A6 order label PDF for printing and attaching to fabric.
    Includes: shop name, customer, garment, delivery, price, QR-like badge.
    Returns path to saved PDF.
    """
    os.makedirs(save_dir, exist_ok=True)
    pdf = FPDF(orientation='L', unit='mm', format='A6')  # landscape A6 = 148x105mm
    pdf.add_page()
    pdf.set_auto_page_break(False)

    # Dark header bar
    pdf.set_fill_color(8, 12, 18)
    pdf.rect(0, 0, 148, 28, 'F')
    pdf.set_draw_color(240, 165, 0)
    pdf.set_line_width(0.5)
    pdf.line(0, 28, 148, 28)

    # Shop / brand name
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(240, 165, 0)
    pdf.set_xy(4, 4)
    pdf.cell(80, 10, 'TAILORCAM ORDER')

    # Order ID
    import random
    order_id = f'TC{random.randint(10000,99999)}'
    if measurement_id:
        order_id = f'TC{measurement_id:05d}'
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(160, 130, 60)
    pdf.set_xy(90, 8)
    pdf.cell(54, 6, f'ID: {order_id}', align='R')

    # Date
    now_str = datetime.datetime.now().strftime('%d %b %Y  %H:%M')
    pdf.set_font('Helvetica', '', 7)
    pdf.set_text_color(130, 140, 150)
    pdf.set_xy(90, 16)
    pdf.cell(54, 5, now_str, align='R')

    # Main content area
    pdf.set_fill_color(12, 20, 30)
    pdf.rect(0, 28, 148, 77, 'F')

    left_col = 6
    right_col = 80

    def row(x_r, y_r, label, value, lbl_col=(120,130,140), val_col=(230,230,240), val_bold=False):
        pdf.set_font('Helvetica', '', 7)
        pdf.set_text_color(*lbl_col)
        pdf.set_xy(x_r, y_r)
        pdf.cell(28, 5, label)
        pdf.set_font('Helvetica', 'B' if val_bold else '', 9)
        pdf.set_text_color(*val_col)
        pdf.set_xy(x_r + 28, y_r - 1)
        pdf.cell(38, 6, str(value)[:22])

    y0 = 32
    row(left_col, y0,      'Customer:',  customer_name[:20],       val_col=(240,240,248), val_bold=True)
    row(left_col, y0+12,   'Garment:',   garment_type[:18],        val_col=(240,165,0),   val_bold=True)
    row(left_col, y0+22,   'Style:',     (style_notes or '--')[:22])
    row(left_col, y0+32,   'Price:',     f'BDT {price}' if price else '--',
        val_col=(40, 195, 50))
    row(left_col, y0+42,   'Delivery:',  delivery_date[:10] or '--',
        val_col=(215, 120, 40) if delivery_date else (130,140,150))

    # Delivery urgency
    if delivery_date:
        try:
            days_left = (datetime.date.fromisoformat(delivery_date[:10]) - datetime.date.today()).days
            if days_left < 0:
                urg = f'OVERDUE {abs(days_left)}d'
                urg_col = (215, 42, 45)
            elif days_left == 0:
                urg = 'DUE TODAY!'
                urg_col = (215, 120, 40)
            else:
                urg = f'{days_left} days remaining'
                urg_col = (40, 195, 50) if days_left > 3 else (215, 120, 40)
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_text_color(*urg_col)
            pdf.set_xy(left_col, y0 + 54)
            pdf.cell(70, 5, urg)
        except Exception:
            pass

    # Right column: QR-like pixel badge (deterministic pattern)
    qr_x, qr_y, qr_sz = right_col, 32, 36
    pdf.set_fill_color(18, 32, 48)
    pdf.rect(qr_x, qr_y, qr_sz, qr_sz, 'F')
    pdf.set_draw_color(240, 165, 0)
    pdf.set_line_width(0.3)
    pdf.rect(qr_x, qr_y, qr_sz, qr_sz, 'D')

    # 6x6 grid pattern seeded from customer name
    seed_v = sum(ord(c)*(i+1) for i,c in enumerate((customer_name + order_id)[:10])) & 0xFFFF
    import random as _r; _r.seed(seed_v)
    cell_mm = qr_sz / 6
    for row_q in range(6):
        for col_q in range(6):
            if row_q in (0,5) and col_q in (0,5): filled = True
            elif _r.random() > 0.45: filled = True
            else: filled = False
            if filled:
                pdf.set_fill_color(240, 165, 0)
                pdf.rect(qr_x + col_q*cell_mm + 0.4,
                         qr_y + row_q*cell_mm + 0.4,
                         cell_mm - 0.8, cell_mm - 0.8, 'F')
    _r.seed(None)

    # Status stages below QR
    stages = ['Received','Cutting','Stitching','Finishing','Ready']
    stage_w = (qr_sz + 40) / len(stages)
    for si, stg in enumerate(stages):
        pdf.set_font('Helvetica', '', 5)
        pdf.set_text_color(100, 110, 120)
        pdf.set_xy(right_col + si * stage_w - 10, qr_y + qr_sz + 2)
        pdf.cell(stage_w, 4, stg, align='C')
        pdf.set_draw_color(60, 75, 90)
        pdf.set_line_width(0.2)
        mid_x = right_col + si * stage_w + stage_w/2 - 10
        pdf.line(mid_x, qr_y + qr_sz + 8, mid_x, qr_y + qr_sz + 11)

    # Bottom border line + footer
    pdf.set_draw_color(240, 165, 0)
    pdf.set_line_width(0.3)
    pdf.line(0, 103, 148, 103)
    pdf.set_font('Helvetica', '', 6)
    pdf.set_text_color(80, 95, 110)
    pdf.set_xy(0, 104.5)
    pdf.cell(148, 4, 'Generated by TailorCam  |  AI Body Measurement System', align='C')

    fname = (f"OrderLabel_{customer_name.replace(' ','_')}_{order_id}_"
             f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    path = os.path.join(save_dir, fname)
    pdf.output(path)
    return path


# ── Pose guidance arrows overlay ──────────────────────────────────────────────
def draw_pose_guidance_arrows(frame, screen_lms_dict, view_name,
                               img_w=None, img_h=None):
    """
    Analyse live landmarks and draw amber guidance arrows/chips when the
    person's pose needs adjustment.

    screen_lms_dict: {landmark_int_id: {'x': px, 'y': px, 'visibility': float}}
    view_name: 'FRONT' | 'LEFT' | 'RIGHT'
    """
    if not screen_lms_dict:
        return

    fh, fw = frame.shape[:2]
    iw = img_w or fw
    ih = img_h or fh

    # landmark ids (MediaPipe)
    LM_NOSE  = 0
    LM_LSH   = 11; LM_RSH   = 12
    LM_LHI   = 23; LM_RHI   = 24
    LM_LWR   = 15; LM_RWR   = 16
    LM_LANK  = 27; LM_RANK  = 28

    def _lm(idx):
        return screen_lms_dict.get(idx)

    def _vis(idx):
        lm = _lm(idx)
        return lm.get('visibility', 0.0) if lm else 0.0

    hints = []

    lsh  = _lm(LM_LSH);   rsh  = _lm(LM_RSH)
    lhi  = _lm(LM_LHI);   rhi  = _lm(LM_RHI)
    lwr  = _lm(LM_LWR);   rwr  = _lm(LM_RWR)
    nose = _lm(LM_NOSE)

    if not lsh or not rsh:
        return

    # Helper: integer pixel coord from a landmark dict value
    def _px(v): return int(round(float(v)))

    lsh_x, lsh_y = _px(lsh['x']), _px(lsh['y'])
    rsh_x, rsh_y = _px(rsh['x']), _px(rsh['y'])

    # ── FRONT view checks ──────────────────────────────────────────────────
    if view_name == 'FRONT':
        sh_span = abs(rsh_x - lsh_x)
        mid_x   = (lsh_x + rsh_x) // 2

        # 1. Distance check: shoulder span too small → person too far
        if sh_span < iw * 0.10:
            hints.append(('STEP CLOSER', mid_x, lsh_y - 30, '>><<'))
        # 2. Distance check: shoulder span too large → too close
        elif sh_span > iw * 0.62:
            hints.append(('STEP BACK', mid_x, lsh_y - 30, '<<>>'))

        # 3. Centering: body_cx should be near iw*0.5
        body_cx = mid_x
        if body_cx < iw * 0.30:
            hints.append(('MOVE RIGHT', body_cx, lsh_y - 45, '->'))
        elif body_cx > iw * 0.70:
            hints.append(('MOVE LEFT', body_cx, lsh_y - 45, '<-'))

        # 4. Arms: wrists should be wider than hips
        if lwr and rwr and lhi and rhi:
            if _vis(LM_LWR) > 0.4 and _vis(LM_RWR) > 0.4:
                lhi_x = _px(lhi['x']); rhi_x = _px(rhi['x'])
                lwr_x = _px(lwr['x']); rwr_x = _px(rwr['x'])
                lwr_y = _px(lwr['y']); rwr_y = _px(rwr['y'])
                if lwr_x > lhi_x - iw * 0.04:
                    hints.append(('ARMS OUT', lwr_x, lwr_y - 20, '<-'))
                if rwr_x < rhi_x + iw * 0.04:
                    hints.append(('ARMS OUT', rwr_x, rwr_y - 20, '->'))

        # 5. Shoulder level: one shoulder much higher
        if abs(lsh_y - rsh_y) > iw * 0.06:
            hints.append(('LEVEL SHOULDERS', mid_x, min(lsh_y, rsh_y) - 20, '--'))

    # ── SIDE view checks ───────────────────────────────────────────────────
    elif view_name in ('LEFT', 'RIGHT'):
        sh_span = abs(rsh_x - lsh_x)
        mid_x   = (lsh_x + rsh_x) // 2

        # Not turned enough: shoulder span still wide
        if sh_span > iw * 0.22:
            direction = 'LEFT' if view_name == 'LEFT' else 'RIGHT'
            hints.append((f'TURN MORE {direction}', mid_x, lsh_y - 40, 'O'))

        # Check centering
        body_cx = mid_x
        if body_cx < iw * 0.25:
            hints.append(('CENTER UP', body_cx, lsh_y - 45, '->'))
        elif body_cx > iw * 0.75:
            hints.append(('CENTER UP', body_cx, lsh_y - 45, '<-'))

    # ── BACK view checks ───────────────────────────────────────────────────
    elif view_name == 'BACK':
        sh_span = abs(rsh_x - lsh_x)
        mid_x   = (lsh_x + rsh_x) // 2

        # Distance checks (same as front)
        if sh_span < iw * 0.10:
            hints.append(('STEP CLOSER', mid_x, lsh_y - 30, '>><<'))
        elif sh_span > iw * 0.62:
            hints.append(('STEP BACK', mid_x, lsh_y - 30, '<<>>'))

        # Centering
        body_cx = mid_x
        if body_cx < iw * 0.30:
            hints.append(('MOVE RIGHT', body_cx, lsh_y - 45, '->'))
        elif body_cx > iw * 0.70:
            hints.append(('MOVE LEFT', body_cx, lsh_y - 45, '<-'))

        # Arms away from body (same check as front)
        if lwr and rwr and lhi and rhi:
            if _vis(LM_LWR) > 0.3 and _vis(LM_RWR) > 0.3:
                lhi_x = _px(lhi['x']); rhi_x = _px(rhi['x'])
                lwr_x = _px(lwr['x']); rwr_x = _px(rwr['x'])
                lwr_y = _px(lwr['y']); rwr_y = _px(rwr['y'])
                if lwr_x > lhi_x - iw * 0.04:
                    hints.append(('ARMS OUT', lwr_x, lwr_y - 20, '<-'))
                if rwr_x < rhi_x + iw * 0.04:
                    hints.append(('ARMS OUT', rwr_x, rwr_y - 20, '->'))

        # Face still visible → not fully turned
        nose_vis = screen_lms_dict.get(0, {}).get('visibility', 0.0) if screen_lms_dict.get(0) else 0.0
        if nose_vis > 0.5:
            hints.append(('TURN AROUND', mid_x, lsh_y - 55, 'O'))

    # ── Render guidance chips ──────────────────────────────────────────────
    for hint_txt, hx, hy, icon in hints[:3]:   # max 3 hints at once
        # Cast ALL coordinates to int — OpenCV requires strict int in Python 3
        hx = int(max(20, min(iw - 20, hx)))
        hy = int(max(50, min(ih - 40, hy)))
        tw = _text_size(hint_txt, FONT, 0.40, 1)[0][0]
        bw = int(tw + 36)
        lx = hx - bw // 2   # left edge of chip
        # Frosted chip
        _alpha(frame, lx, hy - 14, lx + bw, hy + 10, 6, (0, 20, 40), 0.85)
        _rrect_border(frame, lx, hy - 14, lx + bw, hy + 10, 6, GOLD, 2)
        # Icon circle
        cv2.circle(frame, (lx + 14, hy - 2), 8, GOLD, -1)
        _put_text(frame, icon, (lx + 10, hy + 3), FONT, 0.30, DARK, 1)
        # Text
        _text(frame, hint_txt, (lx + 26, hy + 4), GOLD, 0.40, 1)
        # Arrow pointing toward guidance direction
        arr_y = hy + 16
        cv2.arrowedLine(frame, (lx + 14, arr_y), (lx + 14, arr_y + 12),
                        GOLD, 2, tipLength=0.45)


# ── Fabric inventory screen ────────────────────────────────────────────────────
def draw_inventory_screen(frame, items, low_stock_ids=None, mouse_xy=None,
                          selected_category='all'):
    """
    Full-screen fabric & material inventory tracker.
    items: list of dicts from get_inventory() — keys: id, name, category,
           qty, unit, min_qty, colour, price_per
    low_stock_ids: set of item ids where qty < min_qty
    Returns (frame, btns)
    """
    h, w = frame.shape[:2]
    btns = {}
    mx, my = mouse_xy if mouse_xy else (-1, -1)
    low_stock_ids = low_stock_ids or set()

    # ── Background ────────────────────────────────────────────────────────────
    frame[:] = DARK

    # ── Header ────────────────────────────────────────────────────────────────
    _rrect(frame, 0, 0, w, 56, 0, PANEL)
    cv2.line(frame, (0, 56), (w, 56), BORDER, 1)
    lx_v = 24
    pts_v = np.array([[lx_v,8],[lx_v+8,28],[lx_v,48],[lx_v-8,28]], np.int32)
    cv2.fillPoly(frame, [pts_v], GOLD)
    _brand(frame, 30)
    cv2.line(frame, (160, 14), (160, 42), BORDER, 1)
    _text(frame, 'INVENTORY', (174, 36), WHITE, 0.56, 1, FONT_DUP)

    # Low stock badge in header
    n_low = len(low_stock_ids)
    if n_low > 0:
        low_lbl = f'  {n_low} LOW STOCK  '
        lw = _text_size(low_lbl, FONT, 0.38, 1)[0][0] + 4
        _rrect(frame, 360, 14, 360 + lw, 42, 6, (8, 10, 40))
        _rrect_border(frame, 360, 14, 360 + lw, 42, 6, RED, 2)
        _text(frame, low_lbl, (364, 33), RED, 0.38, 1)

    # Back button
    bk_xv = w - 250
    _rrect(frame, bk_xv, 11, bk_xv + 100, 45, 8, CARD)
    _rrect_border(frame, bk_xv, 11, bk_xv + 100, 45, 8, GOLD, 1)
    _text(frame, '<  HOME  H', (bk_xv + 8, 34), GOLD, 0.44, 1)
    btns['back'] = (bk_xv, 11, bk_xv + 100, 45)

    # Add item button
    ai_xv = w - 140
    _rrect(frame, ai_xv, 11, ai_xv + 125, 45, 8, (11, 22, 11))
    _rrect_border(frame, ai_xv, 11, ai_xv + 125, 45, 8, GREEN, 1)
    _text(frame, '+  ADD  A', (ai_xv + 10, 34), GREEN, 0.44, 1)
    btns['add_item'] = (ai_xv, 11, ai_xv + 125, 45)
    btns.update(_page_header(frame, 'Inventory', [('add_item', 'Add item', 'A', 'primary'), ('back', 'Home', 'H', 'normal')],
                             mouse_xy, height=56))
    if n_low:
        _chip(frame, 300, 17, f'{n_low} low on stock', AMBER)

    # ── Category filter tabs ───────────────────────────────────────────────────
    CATEGORIES = ['all', 'fabric', 'thread', 'button', 'other']
    CAT_LABELS  = {'all': 'ALL', 'fabric': 'FABRIC', 'thread': 'THREAD',
                   'button': 'BUTTON', 'other': 'OTHER'}
    tab_x = 14
    tab_y1, tab_y2 = 62, 86
    for cat in CATEGORIES:
        lbl_c = CAT_LABELS[cat]
        tw_c  = _text_size(lbl_c, FONT, 0.38, 1)[0][0] + 20
        active_c = (cat == selected_category)
        bg_c = CARD if active_c else PANEL
        bd_c = GOLD if active_c else BORDER
        _rrect(frame, tab_x, tab_y1, tab_x + tw_c, tab_y2, 6, bg_c)
        _rrect_border(frame, tab_x, tab_y1, tab_x + tw_c, tab_y2, 6, bd_c, 1)
        _text(frame, lbl_c, (tab_x + 10, tab_y2 - 6), GOLD if active_c else GREY, 0.38, 1)
        btns[f'cat_{cat}'] = (tab_x, tab_y1, tab_x + tw_c, tab_y2)
        tab_x += tw_c + 6

    # ── Item grid ─────────────────────────────────────────────────────────────
    grid_x  = 14
    grid_y  = tab_y2 + 10
    col_w   = (w - 42) // 3      # 3-column grid
    col_gap = 7
    card_h  = 96
    card_gap = 8

    # Filter items by category
    if selected_category == 'all':
        shown_items = items
    else:
        shown_items = [it for it in items if it.get('category', 'fabric') == selected_category]

    if not shown_items:
        mid_e = (grid_y + h) // 2
        _centered(frame, 'No items in inventory', mid_e - 16, GREY, 0.58, 1)
        _centered(frame, 'Press  A  to add fabric / materials', mid_e + 18, DIM, 0.42, 1)
    else:
        for idx, item in enumerate(shown_items):
            col_idx = idx % 3
            row_idx = idx // 3
            cx_i = grid_x + col_idx * (col_w + col_gap)
            cy_i = grid_y + row_idx * (card_h + card_gap)
            if cy_i + card_h > h - 10:
                break   # don't draw past bottom

            item_id   = item.get('id')
            is_low    = item_id in low_stock_ids
            is_hov    = (cx_i <= mx <= cx_i + col_w and cy_i <= my <= cy_i + card_h)

            # Glassmorphism card
            _glassmorphism_card(frame, cx_i, cy_i, cx_i + col_w, cy_i + card_h, 10)
            border_col = RED if is_low else (GOLD if is_hov else BORDER)
            _rrect_border(frame, cx_i, cy_i, cx_i + col_w, cy_i + card_h, 10, border_col,
                          2 if (is_low or is_hov) else 1)

            # Item name
            name_i = item.get('name', 'Unknown')[:22]
            _text(frame, name_i, (cx_i + 10, cy_i + 20), WHITE, 0.46, 1)

            # Category chip
            cat_i = item.get('category', 'fabric').upper()[:6]
            cw_i  = _text_size(cat_i, FONT, 0.28, 1)[0][0] + 10
            _rrect(frame, cx_i + col_w - cw_i - 6, cy_i + 4,
                   cx_i + col_w - 6, cy_i + 18, 4, PANEL)
            _text(frame, cat_i,
                  (cx_i + col_w - cw_i - 2, cy_i + 15), TEAL, 0.28, 1)

            # Qty display — large
            qty_i  = item.get('qty', 0)
            unit_i = item.get('unit', 'meters')
            qty_str = f'{qty_i:.1f}' if qty_i != int(qty_i) else str(int(qty_i))
            qw_i = _text_size(qty_str, FONT_DUP, 0.90, 2)[0][0]
            qty_col = RED if is_low else GREEN
            _put_text(frame, qty_str, (cx_i + 12, cy_i + 62),
                        FONT_DUP, 0.90, qty_col, 2)
            _text(frame, unit_i[:8], (cx_i + 14 + qw_i, cy_i + 57), GREY, 0.30, 1)

            # Low stock warning chip
            if is_low:
                min_q = item.get('min_qty', 2)
                warn_s = f'LOW < {min_q:.0f}'
                _rrect(frame, cx_i + 10, cy_i + 68, cx_i + 10 + 70, cy_i + 84, 4,
                       (8, 10, 40))
                _rrect_border(frame, cx_i + 10, cy_i + 68, cx_i + 10 + 70, cy_i + 84, 4,
                              RED, 1)
                _text(frame, warn_s, (cx_i + 14, cy_i + 80), RED, 0.30, 1)

            # Colour swatch
            colour_i = item.get('colour', '')
            if colour_i:
                _text(frame, colour_i[:14], (cx_i + col_w - 80, cy_i + 78), DIM, 0.28, 1)

            # Edit button
            edit_xv = cx_i + col_w - 28
            cv2.arrowedLine(frame, (edit_xv, cy_i + card_h // 2 - 6),
                            (edit_xv + 12, cy_i + card_h // 2 - 6),
                            GOLD if is_hov else BORDER, 1, tipLength=0.55)
            btns[f'item_{idx}'] = (cx_i, cy_i, cx_i + col_w, cy_i + card_h)

    # Footer: total value hint
    if items:
        total_val = sum(it.get('qty', 0) * it.get('price_per', 0) for it in items)
        if total_val > 0:
            ftr = f'Inventory value: BDT {total_val:,.0f}   |   {len(items)} items'
            _centered(frame, ftr, h - 10, DIM, 0.34, 1)

    return frame, btns

