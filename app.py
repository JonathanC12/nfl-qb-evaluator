"""QB Evaluator web app: leaderboard, QB pages, methodology. Reads only from boards/ (committed
snapshots written by update_board.py) -- no nflverse network calls except the one-time team-color
lookup, which is cached.

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
# Palette (validated diverging blue/red pair + status colors; see dataviz skill)
# --------------------------------------------------------------------------- #
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, PAGE_PLANE, GRIDLINE, BORDER = "#fcfcfb", "#f9f9f7", "#e1e0d9", "rgba(11,11,11,0.10)"
POLE_GOOD, POLE_BAD, ROW_HOVER = (0x2A, 0x78, 0xD6), (0xE3, 0x49, 0x48), "#f3f6fb"
STATUS_GOOD, STATUS_WARNING, STATUS_CRITICAL = "#0ca30c", "#fab219", "#d03b3b"

INT_LABELS = {
    "qb_fault_int": ("QB fault", STATUS_CRITICAL),
    "not_qb_fault_int": ("Not QB fault", STATUS_GOOD),
    "dropped_int": ("Dropped INT", STATUS_WARNING),
}

LEADERBOARD_COLS = [
    # key, label, format, colorize, lower_is_good
    ("grade", "Grade", "{:.0f}", True, False),
    ("adj_epa_db", "Adj EPA/play", "{:+.2f}", True, False),
    ("success_rate", "Success %", "{:.0%}", True, False),
    ("total_sack_rate", "Sack %", "{:.1%}", True, True),
    ("int_worthy_rate", "INT-worthy %", "{:.1%}", True, True),
    ("dropbacks", "Dropbacks", "{:.0f}", False, False),
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
def team_colors() -> dict[str, str]:
    try:
        import nflreadpy as nfl
        t = nfl.load_teams().to_pandas()
        return dict(zip(t["team_abbr"], t["team_color"]))
    except Exception:
        return {}


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


def percentile_bg(p: float, lower_is_good: bool = False, max_alpha: float = 0.38) -> str:
    """Diverging background tint: red (bad) <-> white <-> blue (good), subtle by design."""
    if pd.isna(p):
        return SURFACE
    if lower_is_good:
        p = 1 - p
    pole = POLE_GOOD if p >= 0.5 else POLE_BAD
    alpha = abs(p - 0.5) * 2 * max_alpha
    r, g, b = (round(255 * (1 - alpha) + c * alpha) for c in pole)
    return f"#{r:02x}{g:02x}{b:02x}"


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def team_chip(abbr: str, colors: dict[str, str]) -> str:
    bg = colors.get(abbr, INK_MUTED)
    return (f'<span class="chip" style="background:{bg};color:{text_on(bg)}">'
            f'{html.escape(abbr or "")}</span>')


def qb_link(name: str, qb_id: str, season: int) -> str:
    return (f'<a class="qb-link" target="_self" href="?page=qb&qb={qb_id}&season={season}">'
            f'{html.escape(name)}</a>')


def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], section[data-testid="stSidebar"],
.stDeployButton {{ display: none !important; }}

html, body, [class*="css"] {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
body {{ background: {PAGE_PLANE}; }}
.block-container {{ padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1180px; }}

a, a:link, a:visited {{ text-decoration: none; color: inherit; }}

/* top nav */
.qbnav {{
  display: flex; align-items: center; gap: 4px; padding: 14px 2px 10px 2px;
  border-bottom: 1px solid {BORDER}; margin-bottom: 18px; flex-wrap: wrap;
}}
.qbnav .brand {{ font-weight: 700; font-size: 1.05rem; color: {INK}; margin-right: 18px; letter-spacing: -0.01em; }}
.qbnav a, .qbnav a:link, .qbnav a:visited {{
  padding: 7px 12px; border-radius: 6px; color: {INK_SECONDARY}; font-weight: 500; font-size: 0.92rem;
  text-decoration: none !important;
}}
.qbnav a:hover {{ background: {ROW_HOVER}; color: {INK}; }}
.qbnav a.active {{ color: {INK}; font-weight: 600; background: #eef2f9; }}

/* filter row */
.filter-row {{ display: flex; gap: 20px; align-items: flex-end; margin-bottom: 10px; flex-wrap: wrap; }}
.filter-row [data-testid="stVerticalBlock"] > div {{ gap: 0.2rem; }}
label[data-testid="stWidgetLabel"] p {{ font-size: 0.78rem; color: {INK_MUTED}; font-weight: 500; }}

/* leaderboard table */
table.board {{
  width: 100%; border-collapse: collapse; font-size: 0.86rem; background: {SURFACE};
  border: 1px solid {BORDER}; border-radius: 8px; overflow: hidden;
}}
table.board th {{
  text-align: right; font-weight: 600; color: {INK_MUTED}; font-size: 0.72rem;
  text-transform: uppercase; letter-spacing: 0.04em; padding: 9px 12px;
  border-bottom: 1px solid {GRIDLINE}; background: {SURFACE}; position: sticky; top: 0;
}}
table.board th a, table.board th a:link, table.board th a:visited {{
  color: inherit; text-decoration: none !important;
}}
table.board th a:hover {{ color: {INK}; }}
table.board th.l, table.board td.l {{ text-align: left; }}
table.board td {{
  padding: 7px 12px; border-bottom: 1px solid {GRIDLINE}; color: {INK};
  font-variant-numeric: tabular-nums;
}}
table.board td.num {{ text-align: right; }}
table.board tbody tr:hover td {{ background: {ROW_HOVER} !important; }}
table.board td.rank {{ color: {INK_MUTED}; font-variant-numeric: tabular-nums; }}
table.board td.name {{ font-weight: 600; }}
table.board td.grade {{ font-weight: 700; }}
table.board .qb-link {{ color: {INK}; }}
table.board .qb-link:hover {{ text-decoration: underline; }}
.chip {{
  display: inline-block; font-size: 0.68rem; font-weight: 700; padding: 2px 6px;
  border-radius: 4px; letter-spacing: 0.02em; margin-right: 2px;
}}
.table-scroll {{ overflow-x: auto; border-radius: 8px; }}
table.board td.rank.unqualified {{ color: {GRIDLINE}; }}
.low-sample {{
  display: inline-block; font-size: 0.62rem; font-weight: 700; padding: 1px 5px; border-radius: 3px;
  letter-spacing: 0.03em; color: {INK_MUTED}; background: {GRIDLINE}; margin-left: 6px; vertical-align: 1px;
}}

/* board header / view toggle */
.board-head {{ display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap;
  gap: 8px; margin-bottom: 12px; }}
.board-head h2 {{ margin: 0; font-size: 1.15rem; }}
.view-toggle a {{
  padding: 6px 12px; border-radius: 6px; font-size: 0.82rem; font-weight: 500; color: {INK_SECONDARY};
  border: 1px solid {BORDER};
}}
.view-toggle a.active {{ color: {INK}; font-weight: 600; background: #eef2f9; border-color: #eef2f9; }}
.view-toggle a:first-child {{ border-radius: 6px 0 0 6px; border-right: none; }}
.view-toggle a:last-child {{ border-radius: 0 6px 6px 0; }}

.legend {{ display: flex; gap: 16px; font-size: 0.78rem; color: {INK_SECONDARY}; margin: 4px 0 2px 0; }}
.legend .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }}

/* status pill (interceptions) */
.pill {{ display: inline-block; font-size: 0.74rem; font-weight: 600; padding: 3px 9px; border-radius: 20px; color: #fff; }}

/* qb page header */
.qb-hero {{ display: flex; align-items: baseline; gap: 22px; margin: 6px 0 22px 0; flex-wrap: wrap; }}
.qb-hero .grade-num {{ font-size: 3.2rem; font-weight: 700; color: {INK}; line-height: 1; }}
.qb-hero .grade-sub {{ color: {INK_MUTED}; font-size: 0.85rem; }}
.qb-hero h1 {{ font-size: 1.6rem; margin: 0; color: {INK}; font-weight: 700; }}

.section-title {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: {INK_MUTED}; font-weight: 600; margin: 26px 0 8px 0; }}

/* plain table (methodology / component breakdown) */
table.plain {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
table.plain th {{
  text-align: left; font-weight: 600; color: {INK_MUTED}; font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.03em; padding: 7px 10px; border-bottom: 1px solid {GRIDLINE};
}}
table.plain td {{ padding: 7px 10px; border-bottom: 1px solid {GRIDLINE}; font-variant-numeric: tabular-nums; }}
table.plain td.num, table.plain th.num {{ text-align: right; }}

p, li {{ color: {INK_SECONDARY}; line-height: 1.55; }}
h2 {{ color: {INK}; }}

@media (max-width: 480px) {{
  .qb-hero .grade-num {{ font-size: 2.3rem; }}
  table.board {{ font-size: 0.78rem; }}
}}
</style>
""", unsafe_allow_html=True)


def render_nav(active: str) -> None:
    def item(key: str, label: str) -> str:
        cls = "active" if key == active else ""
        return f'<a class="{cls}" target="_self" href="?page={key}">{label}</a>'

    st.markdown(
        '<div class="qbnav"><span class="brand">QB Evaluator</span>'
        + item("leaderboard", "Leaderboard")
        + item("qb", "QB page")
        + item("methodology", "Methodology")
        + "</div>",
        unsafe_allow_html=True,
    )


def html_table(rows: list[str], header_cells: list[str], css_class: str = "plain") -> str:
    head = "".join(header_cells)
    body = "".join(rows)
    return f'<table class="{css_class}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# --------------------------------------------------------------------------- #
# Page: leaderboard
# --------------------------------------------------------------------------- #
def page_leaderboard() -> None:
    seasons = available_seasons()
    if not seasons:
        st.warning("No boards found. Run `python update_board.py` first.")
        return
    season = seasons[-1]
    weeks = available_weeks(season)
    colors = team_colors()
    params = st.query_params
    view = params.get("view", "season")

    c1, c2, c3 = st.columns([1, 1.3, 1.7])
    with c1:
        week = st.selectbox("Week", weeks, index=len(weeks) - 1, key="wk_select")
    board = load_board(season, week) if view == "season" else load_weekly_board(season, week)
    if board.empty:
        st.info("No weekly board for this week yet.")
        return
    default_min_db = int(board["qualifier_dropbacks"].mode().iat[0]) if len(board) else 0
    with c2:
        min_db = st.slider("Min dropbacks", 0, max(int(board["dropbacks"].max()), default_min_db),
                            default_min_db, step=1, key=f"min_db_{view}_{week}")
    teams = sorted(board["team"].dropna().unique())
    with c3:
        team_filter = st.multiselect("Team", teams, key="team_filter", placeholder="All teams")

    def toggle(key: str, label: str) -> str:
        cls = "active" if view == key else ""
        return f'<a class="{cls}" target="_self" href="?page=leaderboard&view={key}">{label}</a>'

    heading = f"Season through Week {week}" if view == "season" else f"Week {week} only"
    st.markdown(
        f'<div class="board-head"><h2>{heading}</h2>'
        f'<div class="view-toggle">{toggle("season", "Season")}{toggle("weekly", "Weekly")}</div></div>',
        unsafe_allow_html=True,
    )
    if view == "weekly":
        st.caption("Single-game grades are noisy and shrunk hard toward league average. "
                   "The season grade (Season tab) is the real ranking.")

    df = board[board["dropbacks"] >= min_db].copy()
    if team_filter:
        df = df[df["team"].isin(team_filter)]

    sort_key = params.get("sort", "grade")
    sort_dir = params.get("dir", "desc")
    sort_map = {"rank": "grade", "qb": "name", "team": "team", **{k: k for k, *_ in LEADERBOARD_COLS}}
    sort_col = sort_map.get(sort_key, "grade")
    df = df.sort_values(sort_col, ascending=(sort_dir == "asc"), na_position="last").reset_index(drop=True)

    # Percentile shading and rank are computed among QUALIFIED rows only (14 dropbacks per team game
    # played, scaled for byes), so a handful of mop-up dropbacks can't distort the scale.
    pct = {k: df[k].where(df["qualified"]).rank(pct=True) for k, *_ in LEADERBOARD_COLS}

    def header(key: str, label: str, align_left: bool = False) -> str:
        new_dir = "asc" if (sort_key == key and sort_dir == "desc") else "desc"
        cls = ' class="l"' if align_left else ""
        return (f'<th{cls}><a target="_self" href="?page=leaderboard&view={view}&sort={key}&dir={new_dir}">'
                f'{label}</a></th>')

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
        cells = (f'<td class="rank l{"" if qualified else " unqualified"}">{rank_text}</td>'
                  f'<td class="name l">{qb_link(r["name"], r["id"], season)}{badge}</td>'
                  f'<td class="l">{team_chip(r["team"], colors)}</td>')
        for key, _, fmt, colorize, lower_good in LEADERBOARD_COLS:
            val = r[key]
            text = fmt.format(val) if pd.notna(val) else "-"
            style = f'background:{percentile_bg(pct[key].iat[i], lower_good)};' if colorize else ""
            weight = "font-weight:700;" if key == "grade" else ""
            cells += f'<td class="num" style="{style}{weight}">{text}</td>'
        rows.append(f"<tr>{cells}</tr>")

    st.markdown(f'<div class="table-scroll">{html_table(rows, [head_cells], "board")}</div>',
                unsafe_allow_html=True)
    st.caption(f"{len(df)} quarterbacks shown, {int(df['qualified'].sum())} qualified "
               f"(at least {default_min_db} dropbacks). Grade and stats for qualified QBs are shrunk "
               "toward league average; see Methodology.")


# --------------------------------------------------------------------------- #
# Page: QB
# --------------------------------------------------------------------------- #
WEEKLY_DOT = "#eb6834"  # categorical slot 2 (orange); season line uses POLE_GOOD (blue, slot 1)


def trend_svg(all_weeks: list[int], cum_by_week: dict[int, float], weekly_by_week: dict[int, float] | None = None,
              width: int = 640, height: int = 170) -> str:
    """Season-to-date grade as a line (+ end label), single-week grades as small dots alongside."""
    pad_l, pad_r, pad_t, pad_b = 8, 54, 16, 24
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    lo, hi = 0, 100
    n = len(all_weeks)
    week_x = {w: pad_l + (plot_w * i / max(n - 1, 1)) for i, w in enumerate(all_weeks)}

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo) / (hi - lo))

    grid = "".join(
        f'<line x1="{pad_l}" y1="{y(v):.1f}" x2="{width - pad_r}" y2="{y(v):.1f}" '
        f'stroke="{GRIDLINE}" stroke-width="1"/>'
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
        blue = f"rgb{POLE_GOOD}"
        pts = [(week_x[w], y(cum_by_week[w])) for w in cum_weeks]
        path = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
        area = (f"M{pts[0][0]:.1f},{y(lo):.1f} " + " ".join(f"L{px:.1f},{py:.1f}" for px, py in pts)
                + f" L{pts[-1][0]:.1f},{y(lo):.1f} Z")
        last_x, last_y = pts[-1]
        line_svg = (f'<path d="{area}" fill="{blue}" opacity="0.08"/>'
                    f'<path d="{path}" fill="none" stroke="{blue}" stroke-width="2" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>'
                    f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4.5" fill="{blue}" '
                    f'stroke="{SURFACE}" stroke-width="2"/>')

    dots = ""
    if weekly_by_week:
        dots = "".join(
            f'<circle cx="{week_x[w]:.1f}" cy="{y(g):.1f}" r="4" fill="{WEEKLY_DOT}" '
            f'stroke="{SURFACE}" stroke-width="1.5"/>'
            for w, g in weekly_by_week.items() if w in week_x
        )

    # Joined with no blank lines: a blank line inside raw HTML breaks Streamlit markdown's HTML-block
    # detection (any piece here can be ""), silently dropping everything after it.
    body = "".join([grid, line_svg, dots, ticks])
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def page_qb() -> None:
    seasons = available_seasons()
    if not seasons:
        st.warning("No boards found. Run `python update_board.py` first.")
        return
    season = int(st.query_params.get("season", seasons[-1]))
    weeks = available_weeks(season)
    latest = load_board(season, weeks[-1])
    colors = team_colors()

    qb_id = st.query_params.get("qb")
    if not qb_id or qb_id not in set(latest["id"]):
        # Every QB with a dropback gets a page: qualified QBs first (by grade), then everyone else
        # (by dropbacks), so a low-sample QB is still reachable from the selector.
        options = latest.sort_values(["qualified", "grade", "dropbacks"], ascending=[False, False, False])
        qb_id = st.selectbox("Choose a QB", options["id"],
                              format_func=lambda i: options.loc[options["id"] == i, "name"].iat[0])
    row = latest[latest["id"] == qb_id].iloc[0]
    qualified = bool(row["qualified"])
    if qualified:
        grade_html = f'<div class="grade-num">{row["grade"]:.0f}</div>'
        n_qualified = int(latest["qualified"].sum())
        rank = int((latest.loc[latest["qualified"], "grade"] > row["grade"]).sum()) + 1
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
    <div>{team_chip(row['team'], colors)} {html.escape(row['team'])} &middot; {row['dropbacks']:.0f} dropbacks
    &middot; through week {weeks[-1]}, {season}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    cum_trend = load_season_trend(season)
    cum_hist = cum_trend[(cum_trend["id"] == qb_id) & cum_trend["grade"].notna()].sort_values("week")
    weekly_trend = load_weekly_trend(season)
    weekly_hist = weekly_trend[(weekly_trend["id"] == qb_id) & weekly_trend["grade"].notna()].sort_values("week")

    if len(cum_hist) or len(weekly_hist):
        st.markdown('<div class="section-title">Grade by week</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="legend">'
            f'<span><span class="dot" style="background:rgb{POLE_GOOD}"></span>Season (cumulative)</span>'
            f'<span><span class="dot" style="background:{WEEKLY_DOT}"></span>Single week</span></div>',
            unsafe_allow_html=True,
        )
        cum_by_week = dict(zip(cum_hist["week"], cum_hist["grade"]))
        weekly_by_week = dict(zip(weekly_hist["week"], weekly_hist["grade"]))
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
            if key not in latest.columns:
                continue
            p = latest[key].where(latest["qualified"]).rank(pct=True)
            val_pct = p[latest["id"] == qb_id].iat[0]
            shown = val_pct if weight >= 0 else 1 - val_pct
            bg = percentile_bg(shown)
            comp_rows.append(
                f'<tr><td>{comp_labels.get(key, key)}</td><td class="num">{weight:+.2f}</td>'
                f'<td class="num">{row[key]:.3f}</td>'
                f'<td class="num" style="background:{bg}">{ordinal(round(val_pct * 100))} pct</td></tr>'
            )
        st.markdown(html_table(comp_rows, ["<th>Component</th><th class='num'>Weight</th>"
                                            "<th class='num'>Value</th><th class='num'>Percentile</th>"]),
                    unsafe_allow_html=True)

    st.markdown('<div class="section-title">Box-score stats (context, not grade drivers)</div>',
                unsafe_allow_html=True)
    context_fields = [
        ("cpoe", "CPOE", "{:+.2f}"), ("passer_rating", "Passer rating", "{:.1f}"),
        ("fault_sack_rate", "QB-fault sack rate", "{:.1%}"),
        ("past_first_read_rate", "Past-first-read rate", "{:.1%}"),
        ("checkdown_rate", "Checkdown rate", "{:.1%}"),
        ("drops", "Drops against", "{:.0f}"),
    ]
    ctx_rows = [f'<tr><td>{label}</td><td class="num">{fmt.format(row[key]) if pd.notna(row[key]) else "-"}</td></tr>'
                for key, label, fmt in context_fields if key in row]
    st.markdown(html_table(ctx_rows, ["<th>Stat</th><th class='num'>Value</th>"]), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Interceptions</div>', unsafe_allow_html=True)
    ints = load_interceptions(season)
    ints = ints[ints["id"] == qb_id].sort_values(["week"]) if len(ints) else ints
    if len(ints) == 0:
        st.markdown("<p>No interceptions this season.</p>", unsafe_allow_html=True)
    else:
        int_rows = []
        for _, r in ints.iterrows():
            label, color = INT_LABELS.get(r["int_bucket"], ("Unlabeled", INK_MUTED))
            desc = html.escape(str(r["desc"]))[:180]
            int_rows.append(
                f'<tr><td>Wk {int(r["week"])}</td><td>{int(r["down"]) if pd.notna(r["down"]) else "-"} '
                f'&amp; {int(r["ydstogo"]) if pd.notna(r["ydstogo"]) else "-"}</td>'
                f'<td><span class="pill" style="background:{color}">{label}</span></td>'
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
    render_nav(page)
    if page == "qb":
        page_qb()
    elif page == "methodology":
        page_methodology()
    else:
        page_leaderboard()


if __name__ == "__main__":
    main()
