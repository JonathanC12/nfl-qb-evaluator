"""QB Evaluator web app: leaderboard, QB pages, methodology. Reads only from boards/ (committed
snapshots written by update_board.py, including team_colors.parquet) -- no network calls of its
own at all, nflreadpy included, so it has nothing to fail or hang on at startup.

Run: streamlit run app.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import qb_eval as q

BOARDS_DIR = Path(__file__).parent / "boards"

# --------------------------------------------------------------------------- #
# Palette. Light theme only (forced in .streamlit/config.toml). Shading is ONE warm sequential hue
# (light peach -> deep orange, deeper = better) on key columns only; dark text stays >= 5.4:1
# contrast on every shade (WCAG AA), so cell text never switches color.
# --------------------------------------------------------------------------- #
INK, INK_SECONDARY, INK_MUTED = "#111827", "#374151", "#6b7280"
PAGE_BG, CARD, DIVIDER, CONTROL_BORDER = "#eef0f4", "#ffffff", "#e5e7eb", "#cfd5df"
NAVY, NAVY_RAISED, NAV_TEXT, NAV_SUBTEXT = "#14213d", "#22325a", "#d6dceb", "#aab4c8"
ACCENT = "#e8692a"  # deep end of the warm scale; also the single-week dots on the QB trend chart
WARM_LO, WARM_HI = (0xFF, 0xF4, 0xEC), (0xE8, 0x69, 0x2A)
ROW_HOVER = "#f7f8fa"
SIDEBAR_W = 232

INT_LABELS = {  # tinted pill backgrounds with dark text, so the label is readable on every pill
    "qb_fault_int": ("QB fault", "#fadcdc"),
    "not_qb_fault_int": ("Not QB fault", "#d9f0dc"),
    "dropped_int": ("Dropped INT", "#fcebc4"),
}

LEADERBOARD_COLS = [
    # key, label, format, shaded (key columns only; higher is always better on these)
    ("grade", "Grade", "{:.0f}", True),
    ("adj_epa_db", "Adj EPA/play", "{:+.2f}", True),
    ("success_rate", "Success %", "{:.0%}", True),
    ("total_sack_rate", "Sack %", "{:.1%}", False),
    ("int_worthy_rate", "INT-worthy %", "{:.1%}", False),
    ("dropbacks", "Dropbacks", "{:.0f}", False),
]


# --------------------------------------------------------------------------- #
# Data access (cached; read-only against boards/)
# --------------------------------------------------------------------------- #
@st.cache_data
def available_seasons() -> list[int]:
    seasons = {int(m.group(1)) for p in BOARDS_DIR.glob("*_wk*.parquet")
               if (m := re.match(r"(\d+)_wk\d+", p.name))}
    return sorted(seasons)


@st.cache_data
def available_weeks(season: int) -> list[int]:
    weeks = {int(m.group(1)) for p in BOARDS_DIR.glob(f"{season}_wk*.parquet")
              if (m := re.match(rf"{season}_wk(\d+)", p.name))}
    return sorted(weeks)


@st.cache_data
def load_board(season: int, week: int) -> pd.DataFrame:
    """Cumulative: season-to-date through this week."""
    return pd.read_parquet(BOARDS_DIR / f"{season}_wk{week}.parquet")


@st.cache_data
def load_weekly_board(season: int, week: int) -> pd.DataFrame:
    """That single week's plays only."""
    path = BOARDS_DIR / f"{season}_wk{week}_weekly.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_season_trend(season: int) -> pd.DataFrame:
    """Every week's cumulative board stacked, for the season grade-trend line."""
    return pd.concat([load_board(season, wk) for wk in available_weeks(season)], ignore_index=True)


@st.cache_data
def load_weekly_trend(season: int) -> pd.DataFrame:
    """Every week's single-week board stacked, for the weekly grade dots."""
    frames = [load_weekly_board(season, wk) for wk in available_weeks(season)]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_interceptions(season: int) -> pd.DataFrame:
    path = BOARDS_DIR / f"{season}_interceptions.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_games(season: int) -> pd.DataFrame:
    """Per-QB game log (box score + single-game grade), written by update_board.py."""
    path = BOARDS_DIR / f"{season}_games.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def team_colors() -> dict[str, str]:
    """Pre-baked by update_board.py (boards/team_colors.parquet) -- the app makes no network calls
    of its own, nflreadpy included, so it has nothing to fail or hang on at startup."""
    path = BOARDS_DIR / "team_colors.parquet"
    if not path.exists():
        return {}
    t = pd.read_parquet(path)
    return dict(zip(t["team_abbr"], t["team_color"]))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def text_on(bg_hex: str) -> str:
    """Black or white text, whichever contrasts better with a background hex color."""
    bg_hex = bg_hex.lstrip("#")
    if len(bg_hex) != 6:
        return INK
    r, g, b = (int(bg_hex[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#111111" if luminance > 0.6 else "#ffffff"


def warm_bg(p: float) -> str:
    """Single-hue sequential shade for a percentile in [0, 1]: light peach (low) -> deep orange (high)."""
    if pd.isna(p):
        return CARD
    r, g, b = (round(lo + (hi - lo) * p) for lo, hi in zip(WARM_LO, WARM_HI))
    return f"#{r:02x}{g:02x}{b:02x}"


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def team_chip(abbr: str, colors: dict[str, str]) -> str:
    bg = colors.get(abbr, INK_MUTED)
    return (f'<span class="chip" style="background:{bg};color:{text_on(bg)}">'
            f'{html.escape(abbr or "")}</span>')


def qb_link(name: str, qb_id: str, season: int, week: int | None = None) -> str:
    wk = f"&week={week}" if week is not None else ""
    return (f'<a class="qb-link" target="_self" href="?page=qb&qb={qb_id}&season={season}{wk}">'
            f'{html.escape(name)}</a>')


def fmt(val, spec: str) -> str:
    return spec.format(val) if pd.notna(val) else "-"


def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], section[data-testid="stSidebar"],
.stDeployButton {{ display: none !important; }}

html, body, [class*="css"] {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: {PAGE_BG} !important; color: {INK}; }}
[data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_W}px; }}

/* main content sits on a white card */
.block-container, [data-testid="stMainBlockContainer"] {{
  background: {CARD}; border: 1px solid {DIVIDER}; border-radius: 10px;
  box-shadow: 0 1px 2px rgba(17,24,39,0.05), 0 4px 14px rgba(17,24,39,0.05);
  max-width: 1160px; margin: 28px auto 40px auto; padding: 26px 30px 30px 30px !important;
}}

a, a:link, a:visited {{ text-decoration: none; color: inherit; }}
p, li {{ color: {INK_SECONDARY}; line-height: 1.55; }}
h1, h2, h3 {{ color: {INK}; }}

/* The CSS and sidebar blocks render as zero-height elements; cancel the flex gap each one adds. */
[data-testid="stElementContainer"]:has(style) {{ margin-bottom: -16px; }}
@media (min-width: 761px) {{
  [data-testid="stElementContainer"]:has(.side) {{ margin-bottom: -16px; }}
}}

/* fixed navy sidebar */
.side {{
  position: fixed; top: 0; left: 0; bottom: 0; width: {SIDEBAR_W}px; z-index: 1000;
  background: {NAVY}; padding: 26px 14px; box-sizing: border-box; overflow-y: auto;
}}
.side .brand-title {{ font-size: 24px; font-weight: 800; color: #ffffff; line-height: 1.15;
  letter-spacing: -0.01em; margin: 0 8px; }}
.side .brand-sub {{ font-size: 0.8rem; color: {NAV_SUBTEXT}; line-height: 1.4; margin: 6px 8px 0 8px; }}
.side nav {{ margin-top: 30px; display: flex; flex-direction: column; gap: 3px; }}
.side nav a, .side nav a:link, .side nav a:visited {{
  display: block; padding: 9px 12px; border-radius: 6px; color: {NAV_TEXT}; font-weight: 500;
  font-size: 0.93rem; border-left: 3px solid transparent;
}}
.side nav a:hover {{ background: {NAVY_RAISED}; color: #ffffff; }}
.side nav a.active {{ background: {NAVY_RAISED}; color: #ffffff; font-weight: 700; border-left-color: {ACCENT}; }}

/* filter row: bold dark labels above white controls */
label[data-testid="stWidgetLabel"] p {{ font-size: 0.82rem; color: {INK}; font-weight: 700; }}
[data-testid="stSelectbox"] [role="group"] {{ background: {CARD} !important; border: 1px solid {CONTROL_BORDER} !important;
  border-radius: 6px; }}
[data-testid="stSelectbox"] [role="group"]:focus-within {{ border-color: {NAVY} !important; }}
[data-testid="stSelectbox"] input {{ color: {INK} !important; -webkit-text-fill-color: {INK}; font-weight: 500; }}

/* leaderboard table */
table.board {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; background: {CARD}; border: none !important; }}
table.board tr, table.plain tr {{ border: none !important; }}
table.board th, table.board td {{ border: none !important; border-bottom: 1px solid {DIVIDER} !important; }}
table.board th {{
  text-align: right; font-weight: 700; color: {INK}; font-size: 0.8rem; line-height: 1.2;
  padding: 8px 10px; background: {CARD}; vertical-align: bottom; white-space: normal;
  border-bottom: 2px solid {INK} !important;
}}
table.board th.num {{ max-width: 86px; }}
table.board th a, table.board th a:link, table.board th a:visited {{ color: {INK}; }}
table.board th a:hover {{ text-decoration: underline; }}
table.board th.l, table.board td.l {{ text-align: left; }}
table.board td {{ padding: 5px 10px; color: {INK}; background: {CARD}; font-variant-numeric: tabular-nums; }}
table.board td.num {{ text-align: right; }}
table.board tbody tr:hover td {{ background: {ROW_HOVER}; }}
table.board td.rank {{ color: {INK_SECONDARY}; }}
table.board td.name {{ font-weight: 600; white-space: nowrap; }}
table.board .qb-link, table.board .qb-link:link, table.board .qb-link:visited {{
  color: {INK}; text-decoration: underline !important; text-decoration-color: {INK_MUTED};
  text-underline-offset: 2px;
}}
table.board .qb-link:hover {{ text-decoration-color: {INK} !important; }}
.chip {{
  display: inline-block; font-size: 0.68rem; font-weight: 700; padding: 2px 6px;
  border-radius: 4px; letter-spacing: 0.02em; margin-right: 2px;
}}
.table-scroll {{ overflow-x: auto; }}
.low-sample {{
  display: inline-block; font-size: 0.62rem; font-weight: 700; padding: 1px 5px; border-radius: 3px;
  letter-spacing: 0.03em; color: {INK_SECONDARY}; background: {DIVIDER}; margin-left: 6px; vertical-align: 1px;
}}
.shade-key {{ display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: {INK_SECONDARY};
  margin: 8px 0 0 0; flex-wrap: wrap; }}
.shade-key .ramp {{ width: 110px; height: 10px; border-radius: 2px; border: 1px solid {DIVIDER};
  background: linear-gradient(90deg, {warm_bg(0)}, {warm_bg(1)}); }}

/* board header / view toggle */
.board-head {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
  gap: 10px; margin: 6px 0 10px 0; }}
.board-head h2 {{ margin: 0; font-size: 1.15rem; font-weight: 700; color: {INK}; }}
.view-toggle {{ display: inline-flex; }}
.view-toggle a, .view-toggle a:link, .view-toggle a:visited {{
  padding: 6px 14px; font-size: 0.84rem; font-weight: 600; color: {INK}; background: {CARD};
  border: 1px solid {CONTROL_BORDER};
}}
.view-toggle a:hover {{ background: {ROW_HOVER}; }}
.view-toggle a.active {{ color: #ffffff; background: {NAVY}; border-color: {NAVY}; }}
.view-toggle a:first-child {{ border-radius: 6px 0 0 6px; border-right: none; }}
.view-toggle a:last-child {{ border-radius: 0 6px 6px 0; }}

.legend {{ display: flex; gap: 16px; font-size: 0.78rem; color: {INK_SECONDARY}; margin: 4px 0 2px 0; }}
.legend .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }}

/* status pill (interceptions) */
.pill {{ display: inline-block; font-size: 0.74rem; font-weight: 600; padding: 3px 9px; border-radius: 20px;
  color: {INK}; white-space: nowrap; }}

/* qb page header */
.qb-hero {{ display: flex; align-items: baseline; gap: 22px; margin: 10px 0 18px 0; flex-wrap: wrap; }}
.qb-hero .grade-num {{ font-size: 3.2rem; font-weight: 800; color: {INK}; line-height: 1; }}
.qb-hero .grade-sub {{ color: {INK_SECONDARY}; font-size: 0.85rem; }}
.qb-hero h1 {{ font-size: 1.6rem; margin: 0; padding: 0; color: {INK}; font-weight: 700; }}
.qb-hero .meta {{ color: {INK_SECONDARY}; }}

.section-title {{ font-size: 0.95rem; color: {INK}; font-weight: 700; margin: 28px 0 4px 0; }}
.section-note {{ font-size: 0.8rem; color: {INK_SECONDARY}; margin: 0 0 10px 0; }}
.context-tag {{ display: inline-block; font-size: 0.66rem; font-weight: 700; letter-spacing: 0.04em;
  padding: 2px 6px; border-radius: 3px; background: {DIVIDER}; color: {INK_SECONDARY}; margin-left: 8px;
  vertical-align: 2px; }}

/* plain tables (stats, game log, components, methodology) */
table.plain {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; border: none !important; }}
table.plain th, table.plain td {{ border: none !important; border-bottom: 1px solid {DIVIDER} !important; }}
table.plain th {{
  text-align: left; font-weight: 700; color: {INK}; font-size: 0.8rem; padding: 6px 10px;
  border-bottom: 2px solid {INK} !important; vertical-align: bottom; background: {CARD};
}}
table.plain td {{ padding: 5px 10px; color: {INK}; font-variant-numeric: tabular-nums; background: {CARD}; }}
table.plain td.num, table.plain th.num {{ text-align: right; }}
.stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }}
.stat-grid h4 {{ font-size: 0.84rem; font-weight: 700; color: {INK}; margin: 0 0 4px 0; padding: 0; }}

/* phone: the sidebar becomes a navy bar across the top of the card */
@media (max-width: 760px) {{
  [data-testid="stAppViewContainer"] {{ padding-left: 0; }}
  .block-container, [data-testid="stMainBlockContainer"] {{
    margin: 0 !important; border-radius: 0; border: none; padding: 0 16px 24px 16px !important;
  }}
  .side {{ position: static; width: auto; margin: 0 -16px 14px -16px; padding: 18px 16px 12px 16px; }}
  .side .brand-title, .side .brand-sub {{ margin-left: 0; margin-right: 0; }}
  .side nav {{ flex-direction: row; flex-wrap: wrap; margin-top: 12px; gap: 4px; }}
  .side nav a, .side nav a:link, .side nav a:visited {{ padding: 6px 10px; border-left: none;
    border-bottom: 3px solid transparent; border-radius: 6px 6px 0 0; }}
  .side nav a.active {{ border-bottom-color: {ACCENT}; }}
  .qb-hero .grade-num {{ font-size: 2.3rem; }}
  table.board {{ font-size: 0.78rem; }}
  table.board th, table.board td {{ padding-left: 6px; padding-right: 6px; }}
  .stat-grid {{ grid-template-columns: 1fr; gap: 18px; }}
  /* filters two per row instead of one long stack */
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: 10px 12px !important; }}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    flex: 1 1 calc(50% - 6px) !important; min-width: calc(50% - 6px) !important; width: calc(50% - 6px) !important;
  }}
}}
</style>
""", unsafe_allow_html=True)


def render_sidebar(active: str) -> None:
    def item(key: str, label: str) -> str:
        cls = "active" if key == active else ""
        return f'<a class="{cls}" target="_self" href="?page={key}">{label}</a>'

    st.markdown(
        '<div class="side">'
        '<div class="brand-title">QB Evaluator</div>'
        '<div class="brand-sub">NFL quarterback grades, updated weekly</div>'
        '<nav>' + item("leaderboard", "Leaderboard") + item("qb", "QB Page")
        + item("methodology", "Methodology") + '</nav></div>',
        unsafe_allow_html=True,
    )


def html_table(rows: list[str], header_cells: list[str], css_class: str = "plain") -> str:
    head = "".join(header_cells)
    body = "".join(rows)
    return f'<table class="{css_class}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def season_week_from_params(seasons: list[int]) -> tuple[int, int | None]:
    """Season and week carried in the query string by plain <a> links (a full page navigation, not a
    Streamlit rerun, which would otherwise reset the selectboxes back to their defaults)."""
    params = st.query_params
    try:
        season = int(params["season"]) if "season" in params else seasons[-1]
    except ValueError:
        season = seasons[-1]
    if season not in seasons:
        season = seasons[-1]
    try:
        week = int(params["week"]) if "week" in params else None
    except ValueError:
        week = None
    return season, week


def week_select(season: int, wanted: int | None, col) -> int:
    weeks = available_weeks(season)
    idx = weeks.index(wanted) if wanted in weeks else len(weeks) - 1
    with col:
        # Keyed by season so switching seasons resets to that season's latest week instead of
        # carrying over a week number that might not exist there.
        week = st.selectbox("Week", weeks, index=idx, key=f"wk_select_{season}")
    st.query_params["week"] = str(week)
    return week


# --------------------------------------------------------------------------- #
# Page: leaderboard
# --------------------------------------------------------------------------- #
def page_leaderboard() -> None:
    seasons = available_seasons()
    if not seasons:
        st.warning("No boards found. Run `python update_board.py` first.")
        return
    colors = team_colors()
    params = st.query_params
    view = params.get("view", "season")
    wanted_season, wanted_week = season_week_from_params(seasons)

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        season = st.selectbox("Season", seasons, index=seasons.index(wanted_season), key="season_select")
    st.query_params["season"] = str(season)
    week = week_select(season, wanted_week if season == wanted_season else None, c1)
    board = load_board(season, week) if view == "season" else load_weekly_board(season, week)
    if board.empty:
        st.info("No weekly board for this week yet.")
        return
    teams = ["All teams"] + sorted(board["team"].dropna().unique())
    with c2:
        team = st.selectbox("Team", teams, key=f"team_filter_{season}")
    qualifier = int(board["qualifier_dropbacks"].mode().iat[0]) if len(board) else 0
    max_db = int(board["dropbacks"].max())
    db_options = sorted({0, qualifier, *[n for n in (25, 50, 100, 150, 200, 300, 400, 500, 600) if n < max_db]})
    with c3:
        min_db = st.selectbox(
            "Min dropbacks", db_options, index=db_options.index(qualifier),
            format_func=lambda n: "Any" if n == 0 else (f"{n}+ (qualified)" if n == qualifier else f"{n}+"),
            key=f"min_db_{season}_{view}_{week}")

    def toggle(key: str, label: str) -> str:
        cls = "active" if view == key else ""
        return (f'<a class="{cls}" target="_self" href="?page=leaderboard&season={season}&week={week}'
                f'&view={key}">{label}</a>')

    heading = f"{season} season through Week {week}" if view == "season" else f"{season}, Week {week} only"
    st.markdown(
        f'<div class="board-head"><h2>{heading}</h2>'
        f'<div class="view-toggle">{toggle("season", "Season")}{toggle("weekly", "Weekly")}</div></div>',
        unsafe_allow_html=True,
    )
    if view == "weekly":
        st.caption("Single-game grades are noisy and shrunk hard toward league average. "
                   "The season grade (Season tab) is the real ranking.")

    df = board[board["dropbacks"] >= min_db].copy()
    if team != "All teams":
        df = df[df["team"] == team]

    sort_key = params.get("sort", "grade")
    sort_dir = params.get("dir", "desc")
    sort_map = {"rank": "grade", "qb": "name", "team": "team", **{k: k for k, *_ in LEADERBOARD_COLS}}
    sort_col = sort_map.get(sort_key, "grade")
    df = df.sort_values(sort_col, ascending=(sort_dir == "asc"), na_position="last").reset_index(drop=True)

    # Shading and rank are computed among QUALIFIED rows only (14 dropbacks per team game played,
    # scaled for byes), so a handful of mop-up dropbacks can't distort the scale.
    pct = {k: df[k].where(df["qualified"]).rank(pct=True) for k, *_, shaded in LEADERBOARD_COLS if shaded}

    def header(key: str, label: str, align_left: bool = False) -> str:
        new_dir = "asc" if (sort_key == key and sort_dir == "desc") else "desc"
        cls = ' class="l"' if align_left else ' class="num"'
        return (f'<th{cls}><a target="_self" href="?page=leaderboard&season={season}&week={week}'
                f'&view={view}&sort={key}&dir={new_dir}">{label}</a></th>')

    head_cells = (header("rank", "Rank", True) + header("qb", "QB", True) + header("team", "Team", True)
                  + "".join(header(k, lbl) for k, lbl, *_ in LEADERBOARD_COLS))

    rows, rank_counter = [], 0
    for i, r in df.iterrows():
        qualified = bool(r["qualified"])
        if qualified:
            rank_counter += 1
            rank_text = str(rank_counter)
        else:
            rank_text = "-"
        badge = '<span class="low-sample">LOW SAMPLE</span>' if not qualified else ""
        cells = (f'<td class="rank l">{rank_text}</td>'
                 f'<td class="name l">{qb_link(r["name"], r["id"], season, week)}{badge}</td>'
                 f'<td class="l">{team_chip(r["team"], colors)}</td>')
        for key, _, spec, shaded in LEADERBOARD_COLS:
            style = f'background:{warm_bg(pct[key].iat[i])};' if shaded and qualified else ""
            weight = "font-weight:700;" if key == "grade" else ""
            cells += f'<td class="num" style="{style}{weight}">{fmt(r[key], spec)}</td>'
        rows.append(f"<tr>{cells}</tr>")

    st.markdown(f'<div class="table-scroll">{html_table(rows, [head_cells], "board")}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="shade-key"><span>Shading on Grade, Adj EPA/play, Success %:</span>'
                '<span>lower</span><span class="ramp"></span><span>higher (better), among qualified QBs</span></div>',
                unsafe_allow_html=True)
    st.caption(f"{len(df)} quarterbacks shown, {int(df['qualified'].sum())} qualified "
               f"(at least {qualifier} dropbacks). Grade and stats for qualified QBs are shrunk "
               "toward league average; see Methodology.")


# --------------------------------------------------------------------------- #
# Page: QB
# --------------------------------------------------------------------------- #
SEASON_LINE = NAVY
WEEKLY_DOT = ACCENT


def trend_svg(all_weeks: list[int], cum_by_week: dict[int, float], weekly_by_week: dict[int, float] | None = None,
              width: int = 640, height: int = 170) -> str:
    """Season-to-date grade as a line, single-week grades as small dots alongside."""
    pad_l, pad_r, pad_t, pad_b = 20, 54, 16, 24
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    lo, hi = 0, 100
    n = len(all_weeks)
    week_x = {w: pad_l + (plot_w * i / max(n - 1, 1)) for i, w in enumerate(all_weeks)}

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo) / (hi - lo))

    grid = "".join(
        f'<line x1="{pad_l}" y1="{y(v):.1f}" x2="{width - pad_r}" y2="{y(v):.1f}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
        f'<text x="{width - pad_r + 8}" y="{y(v):.1f}" dy="3" font-size="10" fill="{INK_MUTED}">{v}</text>'
        for v in (0, 50, 100)
    )
    ticks = "".join(
        f'<text x="{week_x[w]:.1f}" y="{height - 4}" font-size="10" fill="{INK_MUTED}" '
        f'text-anchor="middle">Wk {w}</text>' for w in all_weeks
    )

    # No end-of-line value label: the hero number above the chart already shows the latest grade, and
    # a floating label here would collide with the axis gridline text whenever the grade lands near 0/50/100.
    cum_weeks = [w for w in all_weeks if w in cum_by_week]
    line_svg = ""
    if cum_weeks:
        pts = [(week_x[w], y(cum_by_week[w])) for w in cum_weeks]
        path = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
        area = (f"M{pts[0][0]:.1f},{y(lo):.1f} " + " ".join(f"L{px:.1f},{py:.1f}" for px, py in pts)
                + f" L{pts[-1][0]:.1f},{y(lo):.1f} Z")
        last_x, last_y = pts[-1]
        line_svg = (f'<path d="{area}" fill="{SEASON_LINE}" opacity="0.06"/>'
                    f'<path d="{path}" fill="none" stroke="{SEASON_LINE}" stroke-width="2" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>'
                    f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4.5" fill="{SEASON_LINE}" '
                    f'stroke="{CARD}" stroke-width="2"/>')

    dots = ""
    if weekly_by_week:
        dots = "".join(
            f'<circle cx="{week_x[w]:.1f}" cy="{y(g):.1f}" r="4" fill="{WEEKLY_DOT}" '
            f'stroke="{CARD}" stroke-width="1.5"/>'
            for w, g in weekly_by_week.items() if w in week_x
        )

    # Joined with no blank lines: a blank line inside raw HTML breaks Streamlit markdown's HTML-block
    # detection (any piece here can be ""), silently dropping everything after it.
    body = "".join([grid, line_svg, dots, ticks])
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def stat_table(title: str, fields: list[tuple[str, str, str]], row: pd.Series) -> str:
    rows = "".join(f'<tr><td>{label}</td><td class="num">{fmt(row.get(key), spec)}</td></tr>'
                   for key, label, spec in fields)
    return f'<div><h4>{title}</h4>{html_table([rows], [])}</div>'


PASSING_FIELDS = [
    ("pass_cmp", "Completions", "{:.0f}"), ("pass_att", "Attempts", "{:.0f}"),
    ("comp_pct", "Comp %", "{:.1%}"), ("pass_yds", "Yards", "{:,.0f}"),
    ("pass_ypa", "Yards/attempt", "{:.1f}"), ("pass_td", "TDs", "{:.0f}"),
    ("pass_int", "INTs", "{:.0f}"), ("sacks", "Sacks", "{:.0f}"),
    ("sack_yds", "Sack yards", "{:.0f}"), ("box_passer_rating", "Passer rating", "{:.1f}"),
]
RUSHING_FIELDS = [
    ("rush_att", "Carries", "{:.0f}"), ("rush_yds", "Yards", "{:,.0f}"),
    ("rush_ypc", "Yards/carry", "{:.1f}"), ("rush_td", "Rush TDs", "{:.0f}"),
    ("scrambles", "Scrambles", "{:.0f}"),
]


def render_season_stats(row: pd.Series, season: int, week: int) -> None:
    st.markdown(f'<div class="section-title">Season stats<span class="context-tag">CONTEXT ONLY</span></div>'
                f'<div class="section-note">{season} regular season through Week {week}. Standard box-score '
                f'numbers for reference: they do not drive the grade.</div>', unsafe_allow_html=True)
    if "pass_att" not in row or pd.isna(row["pass_att"]):
        st.markdown("<p>Box-score stats are not available for this board yet.</p>", unsafe_allow_html=True)
        return
    st.markdown('<div class="stat-grid">' + stat_table("Passing", PASSING_FIELDS, row)
                + stat_table("Rushing (scrambles and designed runs)", RUSHING_FIELDS, row) + '</div>',
                unsafe_allow_html=True)


def render_game_log(qb_id: str, season: int, week: int) -> None:
    st.markdown('<div class="section-title">Game log<span class="context-tag">CONTEXT ONLY</span></div>'
                '<div class="section-note">One row per game. Adj EPA/play is that game alone, unshrunk; the '
                'single-game grade is shrunk hard toward average and needs at least 14 dropbacks. '
                'Box-score columns do not drive the grade.</div>', unsafe_allow_html=True)
    games = load_games(season)
    games = games[(games["id"] == qb_id) & (games["week"] <= week)].sort_values("week") if len(games) else games
    if len(games) == 0:
        st.markdown("<p>No games logged yet.</p>", unsafe_allow_html=True)
        return
    rows = []
    for _, g in games.iterrows():
        opp = ("vs " if g["home"] else "@ ") + html.escape(str(g["opponent"]))
        grade_style = f' style="background:{warm_bg(g["grade"] / 100)};font-weight:700"' if pd.notna(g["grade"]) else ""
        rows.append(
            f'<tr><td>{int(g["week"])}</td><td>{opp}</td>'
            f'<td class="num">{int(g["pass_cmp"])}/{int(g["pass_att"])}</td>'
            f'<td class="num">{int(g["pass_yds"])}</td><td class="num">{int(g["pass_td"])}</td>'
            f'<td class="num">{int(g["pass_int"])}</td><td class="num">{int(g["rush_yds"])}</td>'
            f'<td class="num">{fmt(g["adj_epa_db"], "{:+.2f}")}</td>'
            f'<td class="num"{grade_style}>{fmt(g["grade"], "{:.0f}")}</td></tr>'
        )
    head = ("<th>Week</th><th>Opp</th><th class='num'>Comp/Att</th><th class='num'>Pass yds</th>"
            "<th class='num'>TD</th><th class='num'>INT</th><th class='num'>Rush yds</th>"
            "<th class='num'>Adj EPA/play</th><th class='num'>Game grade</th>")
    st.markdown(f'<div class="table-scroll">{html_table(rows, [head])}</div>', unsafe_allow_html=True)


def page_qb() -> None:
    seasons = available_seasons()
    if not seasons:
        st.warning("No boards found. Run `python update_board.py` first.")
        return
    colors = team_colors()
    wanted_season, wanted_week = season_week_from_params(seasons)

    c0, c1, c2 = st.columns([2, 1, 1])
    with c1:
        season = st.selectbox("Season", seasons, index=seasons.index(wanted_season), key="qb_season_select")
    st.query_params["season"] = str(season)
    week = week_select(season, wanted_week if season == wanted_season else None, c2)
    board = load_board(season, week)

    # Every QB with a dropback gets a page: qualified QBs first (by grade), then everyone else
    # (by dropbacks), so a low-sample QB is still reachable from the selector.
    options = board.sort_values(["qualified", "grade", "dropbacks"], ascending=[False, False, False])
    ids = list(options["id"])
    wanted_qb = st.query_params.get("qb")
    names = dict(zip(options["id"], options["name"]))
    with c0:
        qb_id = st.selectbox("Quarterback", ids, index=ids.index(wanted_qb) if wanted_qb in ids else 0,
                             format_func=lambda i: f'{names[i]} ({board.loc[board["id"] == i, "team"].iat[0]})',
                             key=f"qb_select_{season}_{week}")
    st.query_params["qb"] = qb_id

    row = board[board["id"] == qb_id].iloc[0]
    qualified = bool(row["qualified"])
    if qualified:
        grade_html = f'<div class="grade-num">{row["grade"]:.0f}</div>'
        n_qualified = int(board["qualified"].sum())
        rank = int((board.loc[board["qualified"], "grade"] > row["grade"]).sum()) + 1
        sub_html = f'<div class="grade-sub">Rank {rank} of {n_qualified} qualified</div>'
    else:
        grade_html = '<div class="grade-num" style="font-size:1.8rem">Low sample</div>'
        need = int(row["qualifier_dropbacks"] - row["dropbacks"])
        sub_html = f'<div class="grade-sub">Needs {max(need, 0)} more dropbacks to qualify</div>'

    st.markdown(f"""
<div class="qb-hero">
  <div>
    {grade_html}
    {sub_html}
  </div>
  <div>
    <h1>{html.escape(row['name'])}</h1>
    <div class="meta">{team_chip(row['team'], colors)} {html.escape(row['team'])} &middot; {row['dropbacks']:.0f} dropbacks
    &middot; through week {week}, {season}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    weeks = [w for w in available_weeks(season) if w <= week]
    cum_trend = load_season_trend(season)
    cum_hist = cum_trend[(cum_trend["id"] == qb_id) & cum_trend["grade"].notna() & (cum_trend["week"] <= week)]
    weekly_trend = load_weekly_trend(season)
    weekly_hist = (weekly_trend[(weekly_trend["id"] == qb_id) & weekly_trend["grade"].notna()
                                & (weekly_trend["week"] <= week)] if len(weekly_trend) else weekly_trend)

    if len(cum_hist) or len(weekly_hist):
        st.markdown('<div class="section-title">Grade by week</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="legend">'
            f'<span><span class="dot" style="background:{SEASON_LINE}"></span>Season (cumulative)</span>'
            f'<span><span class="dot" style="background:{WEEKLY_DOT}"></span>Single week</span></div>',
            unsafe_allow_html=True,
        )
        cum_by_week = dict(zip(cum_hist["week"], cum_hist["grade"]))
        weekly_by_week = dict(zip(weekly_hist["week"], weekly_hist["grade"])) if len(weekly_hist) else {}
        st.markdown(trend_svg(weeks, cum_by_week, weekly_by_week), unsafe_allow_html=True)
        st.caption("Single-week grades are shrunk hard toward league average and still bounce around a lot. "
                   "The season (cumulative) line is the real ranking.")

    st.markdown('<div class="section-title">Grade components</div>', unsafe_allow_html=True)
    if not qualified:
        st.markdown("<p>Not enough dropbacks yet to qualify for a graded breakdown.</p>", unsafe_allow_html=True)
    else:
        comp_labels = {
            "adj_epa_db": "Adj EPA/dropback", "success_rate": "Success rate",
            "int_worthy_rate": "INT-worthy rate", "total_sack_rate": "Total sack rate",
            "rush_epa_db": "Rush EPA/dropback",
        }
        comp_rows = []
        for key, weight in q.CONFIG["weights"].items():
            if key not in board.columns:
                continue
            p = board[key].where(board["qualified"]).rank(pct=True)
            val_pct = p[board["id"] == qb_id].iat[0]
            # Shaded by how GOOD the value is (so a low sack rate is a deep shade), not by raw size.
            good = val_pct if weight >= 0 else 1 - val_pct
            comp_rows.append(
                f'<tr><td>{comp_labels.get(key, key)}</td><td class="num">{weight:+.2f}</td>'
                f'<td class="num">{row[key]:.3f}</td>'
                f'<td class="num" style="background:{warm_bg(good)}">{ordinal(round(val_pct * 100))} pct</td></tr>'
            )
        st.markdown(html_table(comp_rows, ["<th>Component</th><th class='num'>Weight</th>"
                                            "<th class='num'>Value</th><th class='num'>Percentile</th>"]),
                    unsafe_allow_html=True)

    render_season_stats(row, season, week)
    render_game_log(qb_id, season, week)

    st.markdown('<div class="section-title">Charting context<span class="context-tag">CONTEXT ONLY</span></div>'
                '<div class="section-note">FTN-charted process stats. Tested as grade inputs and left out; '
                'see Methodology.</div>', unsafe_allow_html=True)
    context_fields = [
        ("cpoe", "CPOE", "{:+.2f}"),
        ("fault_sack_rate", "QB-fault sack rate", "{:.1%}"),
        ("past_first_read_rate", "Past-first-read rate", "{:.1%}"),
        ("checkdown_rate", "Checkdown rate", "{:.1%}"),
        ("drops", "Drops against", "{:.0f}"),
    ]
    ctx_rows = [f'<tr><td>{label}</td><td class="num">{fmt(row[key], spec)}</td></tr>'
                for key, label, spec in context_fields if key in row]
    st.markdown(html_table(ctx_rows, ["<th>Stat</th><th class='num'>Value</th>"]), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Interceptions</div>', unsafe_allow_html=True)
    ints = load_interceptions(season)
    ints = ints[(ints["id"] == qb_id) & (ints["week"] <= week)].sort_values(["week"]) if len(ints) else ints
    if len(ints) == 0:
        st.markdown("<p>No interceptions this season.</p>", unsafe_allow_html=True)
    else:
        int_rows = []
        for _, r in ints.iterrows():
            label, tint = INT_LABELS.get(r["int_bucket"], ("Unlabeled", DIVIDER))
            desc = html.escape(str(r["desc"]))[:180]
            int_rows.append(
                f'<tr><td>Wk {int(r["week"])}</td><td>{int(r["down"]) if pd.notna(r["down"]) else "-"} '
                f'&amp; {int(r["ydstogo"]) if pd.notna(r["ydstogo"]) else "-"}</td>'
                f'<td><span class="pill" style="background:{tint}">{label}</span></td>'
                f'<td style="color:{INK_SECONDARY}">{desc}</td></tr>'
            )
        st.markdown(html_table(int_rows, ["<th>Week</th><th>Down &amp; dist</th><th>Fault</th><th>Play</th>"]),
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page: methodology
# --------------------------------------------------------------------------- #
def page_methodology() -> None:
    st.markdown("## Methodology")
    st.markdown(f"""
<p>The grade starts from each quarterback's EPA (expected points added) per dropback, then adjusts it for
<strong>who was at fault</strong> on interceptions, using FTN charting layered on top of nflverse play-by-play.
Every change to the model is backtested before it is adopted: a metric has to predict a quarterback's EPA in
games it has not seen, not just describe the games it has already played.</p>

<p>Two tests, run on the 2022 to 2025 regular seasons: <strong>split-half</strong> (odd weeks predict even weeks,
within a season) and <strong>year over year</strong> (one season predicts the next, same quarterback). The
baseline every metric has to beat is plain raw EPA per dropback.</p>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">What is in the grade (v3)</div>', unsafe_allow_html=True)
    rows = [f'<tr><td>{k.replace("_", " ")}</td><td class="num">{w:+.2f}</td></tr>'
            for k, w in q.CONFIG["weights"].items()]
    st.markdown(html_table(rows, ["<th>Component</th><th class='num'>Weight</th>"]), unsafe_allow_html=True)
    st.markdown("""
<p style="margin-top:10px">Adjusted EPA uses only one fault adjustment: interceptions that were not an
interception-worthy throw (tipped, receiver-caused) are scored as an incompletion instead of a pick, and
interception-worthy throws that were <em>not</em> caught ("dropped interceptions") are charged their expected
cost. A garbage-time filter, full credit for receiver drops, and replacing actual YAC with expected YAC were
all tried and made the grade worse; they stay out. See the ablation and YAC sections of the backtest notebook
for the numbers.</p>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">What v3 added, and what it tried that did not work</div>',
                unsafe_allow_html=True)
    st.markdown("""
<p>Three additions were tested: success rate (share of dropbacks with positive EPA), total sack rate in place
of FTN's charted "QB-fault" sack rate, and read-progression rates from FTN's <code>read_thrown</code> field
(share of attempts that reached a second-or-later read, checkdown rate, EPA on second-or-later-read throws).</p>
""", unsafe_allow_html=True)
    comp = pd.DataFrame([
        {"metric": "Success rate (EPA>0)", "split_half": 0.521, "yoy": 0.471, "verdict": "Added to grade"},
        {"metric": "Total sack rate", "split_half": -0.363, "yoy": -0.436, "verdict": "Added to grade (replaces QB-fault sack rate)"},
        {"metric": "QB-fault sack rate (old)", "split_half": -0.142, "yoy": -0.346, "verdict": "Removed from grade"},
        {"metric": "Past-first-read rate", "split_half": 0.053, "yoy": -0.241, "verdict": "Not used: sign flips"},
        {"metric": "Checkdown rate", "split_half": -0.093, "yoy": -0.134, "verdict": "Not used: weak"},
        {"metric": "EPA on 2nd+ read throws", "split_half": float("nan"), "yoy": 0.189, "verdict": "Not used: too sparse, unstable"},
    ])
    rows = [f'<tr><td>{r.metric}</td><td class="num">{r.split_half:+.3f}</td><td class="num">{r.yoy:+.3f}</td>'
            f'<td>{r.verdict}</td></tr>' if pd.notna(r.split_half) else
            f'<tr><td>{r.metric}</td><td class="num">n/a</td><td class="num">{r.yoy:+.3f}</td><td>{r.verdict}</td></tr>'
            for r in comp.itertuples()]
    st.markdown(html_table(rows, ["<th>Metric</th><th class='num'>Split-half r</th>"
                                   "<th class='num'>Year-over-year r</th><th>Verdict</th>"]),
                unsafe_allow_html=True)

    st.markdown("""
<p style="margin-top:14px">Read-progression metrics were dropped from the grade: FTN's <code>read_thrown</code>
charting is too noisy at the QB-week level to carry weight, and one of the three metrics flips sign between
the two tests, which is a sign of noise rather than signal. They are still shown on QB pages as context.
One data note: the published nflreadr data dictionary for <code>read_thrown</code> has its <code>"0"</code> and
<code>"1"</code> codes backwards from what the charting data actually shows; the codes used here (<code>1</code>
= first read, <code>2</code> = second-or-later read, <code>CHK</code> = checkdown, <code>DES</code> =
designed/screen, <code>SD</code> = scramble drill, <code>0</code>/missing = no read charted) were confirmed
directly against sack rate, completion rate, and air yards by code, not taken from the dictionary page.</p>

<p>The resulting composite beats the previous one on both tests (split-half r 0.505 &rarr; 0.538, year-over-year
0.457 &rarr; 0.493) and now edges past raw EPA year over year, though it still trails raw EPA in split-half.
None of these gains clear statistical significance with four seasons of data: they are directional, chosen
after looking at the backtest (like the very first interception-fault adjustment was), and should be read as
"promising, unproven." The full numbers, including bootstrap confidence intervals, are in
<code>qb_backtest.ipynb</code>.</p>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Other things that were tried and did not help</div>',
                unsafe_allow_html=True)
    st.markdown("""
<ul>
<li>Full credit for every receiver drop (too generous; some drops are on badly placed balls).</li>
<li>A garbage-time win-probability filter (throws away 20 to 30 percent of plays for little noise reduction).</li>
<li>Replacing actual yards-after-catch with expected YAC (makes the grade less stable year over year, not more).</li>
<li>A mixed model separating quarterback from defense and offense: removing the surrounding offense's effect
hurts, since a starter takes nearly every snap and the model cannot tell a good quarterback from a good offense.</li>
<li>CPOE, completion percentage, yards, touchdowns, and ANY/A as grade components: no measurable lift, and
adding all of them together made the grade worse.</li>
</ul>
<p>For quarterbacks who changed teams, no metric in this project predicted their next season with any
reliability (n=14, correlation near zero for every candidate). A grade built on EPA mostly describes a
quarterback within his situation, not independent of it.</p>

<p><strong>2026 is held out.</strong> Nothing in this model was tuned on 2026 results; the plan is to score
this model's predictions against the 2026 season once it is complete.</p>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="QB Evaluator", layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    page = st.query_params.get("page", "leaderboard")
    render_sidebar(page)
    if page == "qb":
        page_qb()
    elif page == "methodology":
        page_methodology()
    else:
        page_leaderboard()


if __name__ == "__main__":
    main()
