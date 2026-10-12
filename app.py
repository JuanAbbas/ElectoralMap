"""
Swing State — Electoral College & Midterm Simulator
Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import election_data as ed
import model

st.set_page_config(page_title="Swing State", page_icon="🗳️", layout="wide")

# ----------------------------------------------------------------- styling
D_BLUE, R_RED, NEUTRAL = "#2E5EAA", "#C23B33", "#E7E3EC"
INK, MUTED, GRID = "#1D2230", "#6B7080", "rgba(120,125,140,0.18)"
PROB_SCALE = [
    [0.00, "#8E2219"], [0.15, "#D05A4F"], [0.35, "#EFC2BA"], [0.50, NEUTRAL],
    [0.65, "#BCCDEE"], [0.85, "#5A83CC"], [1.00, "#1C4489"],
]
RATING_COLORS = {"Safe D": "#1C4489", "Likely D": "#4C78C6", "Lean D": "#9DB7E6",
                 "Toss-up": "#CFC8D9", "Lean R": "#EBA79E", "Likely R": "#CF5A4E", "Safe R": "#8E2219"}

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stText, label, p, li { font-family: 'Public Sans', system-ui, sans-serif; }
h1, h2, h3, .ss-display { font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2rem; max-width: 1400px; }
.ss-hero { display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between; gap:1rem;
           border-bottom: 2px solid #1D2230; padding-bottom: .6rem; margin-bottom: 1rem; }
.ss-hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0; line-height: 1; }
.ss-hero .tag { color:#6B7080; font-size:.95rem; margin-top:.35rem; }
.ss-count { font-family:'IBM Plex Mono', monospace; font-size:.85rem; color:#6B7080; text-align:right; }
.ss-count b { color:#1D2230; font-size:1.05rem; }
.ss-big { display:grid; grid-template-columns: 1fr auto 1fr; align-items:end; gap:1rem; margin:.4rem 0 .2rem; }
.ss-side .lab { font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; color:#6B7080; font-weight:600; }
.ss-side .num { font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:3.1rem; line-height:1; font-variant-numeric: tabular-nums; }
.ss-side .sub { font-family:'IBM Plex Mono', monospace; font-size:.85rem; color:#6B7080; }
.ss-mid { text-align:center; font-family:'IBM Plex Mono', monospace; font-size:.8rem; color:#6B7080; padding-bottom:.6rem; }
.ss-r { text-align:right; }
.ss-chip { display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.75rem; font-weight:600;
           font-family:'IBM Plex Mono', monospace; }
div[data-testid="stMetricValue"] { font-family:'Bricolage Grotesque', sans-serif; font-variant-numeric: tabular-nums; }
.stTabs [data-baseweb="tab"] { font-family:'Bricolage Grotesque', sans-serif; font-size:1.05rem; }
.ss-note { font-size:.82rem; color:#6B7080; }
</style>
""",
    unsafe_allow_html=True,
)

TODAY = date.today()
DAYS_MID = (date.fromisoformat(ed.ELECTION_DAY_MIDTERM) - TODAY).days
DAYS_PRES = (date.fromisoformat(ed.ELECTION_DAY_PRES) - TODAY).days


def fmt_m(m: float) -> str:
    if abs(m) < 0.05:
        return "Even"
    return f"D+{m:.1f}" if m > 0 else f"R+{-m:.1f}"


def pct(p: float) -> str:
    if p > 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{p*100:.0f}%"


def base_layout(fig, h=320, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Public Sans, sans-serif", size=12),
                      showlegend=False, **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ------------------------------------------------------------ session state
ss = st.session_state
ss.setdefault("adj_pres", {})
ss.setdefault("locks_pres", {})
ss.setdefault("locks_sen", {})
ss.setdefault("generic_polls", pd.DataFrame(ed.SEED_GENERIC_POLLS))
ss.setdefault("pres_polls", pd.DataFrame(ed.SEED_PRES_POLLS))
ss.setdefault("senate_races", model.senate_table())
ss.setdefault("sel_state", "PA")
ss.setdefault("nat_pres", ed.NATIONAL_2024_PRES)

# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Model engine")
    n_sims = st.select_slider("Simulations", [2000, 5000, 10000, 20000, 40000], value=20000,
                              help="More simulations = smoother probabilities, slower updates.")
    elastic = st.toggle("Elastic swing", value=True,
                        help="On: swingy states (NH, ME, WI) move more than polarised ones (MS, DC). Off: uniform swing.")
    sigma_reg = st.slider("Regional error (pts)", 0.0, 5.0, 2.0, 0.25,
                          help="Correlated miss shared by states in the same region (Rust Belt, Sun Belt…).")
    sigma_state = st.slider("State error (pts)", 0.5, 7.0, 3.0, 0.25,
                            help="Independent state-level noise.")
    half_life = st.slider("Poll half-life (days)", 3, 60, 14,
                          help="A poll this many days old counts half as much as a fresh one.")
    st.divider()
    st.markdown("### House curve")
    swing_ratio = st.slider("Seats per point", 2.0, 8.0, 4.5, 0.25,
                            help="Seats Democrats gain per 1-pt shift in the generic ballot near the centre.")
    map_bias = st.slider("Break-even generic margin", -3.0, 5.0, 1.5, 0.25,
                         help="Generic ballot margin (D−R) at which the House splits evenly. Positive = map favours GOP.")
    st.divider()
    if st.button("Reset all scenarios", width="stretch"):
        for k in ["adj_pres", "locks_pres", "locks_sen"]:
            ss[k] = {}
        ss["senate_races"] = model.senate_table()
        ss["nat_pres"] = ed.NATIONAL_2024_PRES
        st.rerun()

# ------------------------------------------------------------------- hero
st.markdown(
    f"""
<div class="ss-hero">
  <div><h1>Swing State</h1>
  <div class="tag">Electoral College &amp; midterm simulator · polling in, {n_sims:,} correlated elections out</div></div>
  <div class="ss-count">Midterms <b>{max(DAYS_MID,0)}</b> days · Nov 3, 2026<br>Presidential <b>{DAYS_PRES}</b> days · Nov 7, 2028</div>
</div>
""",
    unsafe_allow_html=True,
)

gen_avg, gen_eff, gen_table = model.poll_average(ss.generic_polls, TODAY, half_life)
pres_avg, pres_eff, pres_table = model.poll_average(ss.pres_polls, TODAY, half_life)


# ---------------------------------------------------------- cached sims
@st.cache_data(show_spinner=False)
def run_pres(national, sigma_nat, sigma_reg, sigma_state, elastic, adj_t, lock_t, n):
    return model.simulate_president(national, sigma_nat, sigma_reg, sigma_state, elastic,
                                    dict(adj_t), dict(lock_t), n)


@st.cache_data(show_spinner=False)
def run_senate(races_json, generic, sigma_nat, sigma_reg, sigma_state, lock_t, n):
    races = pd.read_json(io.StringIO(races_json))
    return model.simulate_senate(races, generic, sigma_nat, sigma_reg, sigma_state + 1.0, dict(lock_t), n)


@st.cache_data(show_spinner=False)
def run_house(generic, sigma_nat, swing_ratio, map_bias, n):
    return model.simulate_house(generic, sigma_nat, swing_ratio, map_bias, n_sims=n)


# ------------------------------------------------------------------ figures
def us_map(df, loc_col, p_col, hover, height=430, missing=None):
    fig = go.Figure()
    if missing is not None and len(missing):
        fig.add_trace(go.Choropleth(locations=missing, z=[0] * len(missing), locationmode="USA-states",
                                    colorscale=[[0, "rgba(160,160,170,0.22)"], [1, "rgba(160,160,170,0.22)"]],
                                    showscale=False, marker_line_color="white", marker_line_width=0.8,
                                    hoverinfo="location"))
    fig.add_trace(go.Choropleth(
        locations=df[loc_col], z=df[p_col], locationmode="USA-states", zmin=0, zmax=1,
        colorscale=PROB_SCALE, marker_line_color="white", marker_line_width=0.9,
        text=hover, hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title=dict(text="P(Dem)", side="top"), tickformat=".0%", thickness=10, len=0.55,
                      x=0.98, tickvals=[0, .25, .5, .75, 1], orientation="v"),
    ))
    fig.add_trace(go.Scattergeo(locations=df[loc_col], locationmode="USA-states", text=df[loc_col],
                                mode="text", textfont=dict(size=9, color="#1D2230", family="IBM Plex Mono"),
                                hoverinfo="skip"))
    fig.update_geos(scope="usa", bgcolor="rgba(0,0,0,0)", showlakes=False, lakecolor="rgba(0,0,0,0)")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      dragmode=False, clickmode="event+select")
    return fig


def bar_270(units):
    u = units.copy()
    u["rating"] = u["p_d"].apply(model.rating)
    agg = u.groupby("rating")["ev"].sum().reindex(model.RATING_ORDER).fillna(0)
    fig = go.Figure()
    for r in model.RATING_ORDER:
        if agg[r] > 0:
            fig.add_trace(go.Bar(x=[agg[r]], y=[""], orientation="h", name=r, marker_color=RATING_COLORS[r],
                                 text=f"{r}<br>{int(agg[r])}" if agg[r] >= 25 else int(agg[r]),
                                 textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color="white" if "Safe" in r or "Likely" in r else INK, size=11),
                                 hovertemplate=f"{r}: %{{x}} EV<extra></extra>"))
    fig.add_vline(x=270, line_color=INK, line_width=2)
    fig.add_annotation(x=270, y=0.62, yref="paper", text="270 to win", showarrow=False, yshift=28,
                       font=dict(family="IBM Plex Mono", size=11))
    base_layout(fig, h=110, barmode="stack")
    fig.update_layout(margin=dict(l=0, r=0, t=26, b=0))
    fig.update_xaxes(range=[0, 538], showgrid=False, showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def ev_hist(ev_d):
    vals, counts = np.unique(ev_d, return_counts=True)
    p = counts / counts.sum()
    colors = [D_BLUE if v >= 270 else (MUTED if v == 269 else R_RED) for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=p, marker_color=colors, marker_line_width=0,
                           hovertemplate="Dem %{x} EV: %{y:.2%}<extra></extra>"))
    fig.add_vline(x=269.5, line_dash="dot", line_color=INK)
    base_layout(fig, h=300, title=dict(text="Distribution of Democratic electoral votes", font=dict(size=14)))
    fig.update_yaxes(tickformat=".1%")
    fig.update_xaxes(title="Democratic electoral votes", range=[max(vals.min() - 10, 0), min(vals.max() + 10, 538)])
    return fig


def snake(units):
    u = units.sort_values("mean_margin", ascending=False).reset_index(drop=True)
    u["start"] = u["ev"].cumsum() - u["ev"]
    fig = go.Figure(go.Bar(
        x=u["ev"], base=u["start"], y=[""] * len(u), orientation="h",
        marker=dict(color=u["p_d"], colorscale=PROB_SCALE, cmin=0, cmax=1, line=dict(color="white", width=1)),
        text=[x if e >= 6 else "" for x, e in zip(u["unit"], u["ev"])], textposition="inside",
        insidetextanchor="middle", textfont=dict(size=9, family="IBM Plex Mono"),
        customdata=np.stack([u["name"], u["ev"], u["mean_margin"].map(fmt_m), u["p_d"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]} EV<br>Mean %{customdata[2]} · P(D) %{customdata[3]:.0%}<extra></extra>",
    ))
    fig.add_vline(x=270, line_color=INK, line_width=2)
    base_layout(fig, h=120, title=dict(text="The snake: every EV unit, ordered from most Democratic to most Republican",
                                        font=dict(size=13)))
    fig.update_xaxes(range=[0, 538], showgrid=False, tickvals=[0, 100, 200, 270, 338, 438, 538])
    fig.update_yaxes(showticklabels=False)
    return fig


def tipping_chart(series, names, title, n=10):
    s = series.head(n)[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=[names.get(k, k) for k in s.index], orientation="h",
                           marker_color="#6E5BA8", text=[f"{v:.1%}" for v in s.values], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    base_layout(fig, h=320, title=dict(text=title, font=dict(size=14)))
    fig.update_xaxes(tickformat=".0%", range=[0, max(s.values.max() * 1.25, 0.05)])
    return fig


def big_numbers(left_lab, left_num, left_sub, mid, right_lab, right_num, right_sub):
    st.markdown(f"""
<div class="ss-big">
  <div class="ss-side"><div class="lab" style="color:{D_BLUE}">{left_lab}</div>
    <div class="num" style="color:{D_BLUE}">{left_num}</div><div class="sub">{left_sub}</div></div>
  <div class="ss-mid">{mid}</div>
  <div class="ss-side ss-r"><div class="lab" style="color:{R_RED}">{right_lab}</div>
    <div class="num" style="color:{R_RED}">{right_num}</div><div class="sub">{right_sub}</div></div>
</div>""", unsafe_allow_html=True)


def selected_from_event(event, valid):
    try:
        pts = event.selection.points if hasattr(event, "selection") else event["selection"]["points"]
        for p in pts:
            loc = p.get("location")
            if loc in valid:
                return loc
    except Exception:
        pass
    return None


# ===================================================================== TABS
tab_p, tab_m, tab_polls = st.tabs(["2028 Presidential", "2026 Midterms", "Polls & method"])

# ------------------------------------------------------------ PRESIDENTIAL
with tab_p:
    c1, c2, c3 = st.columns([1.3, 1.3, 1])
    with c1:
        src = st.radio("National environment from", ["Manual", "Poll average", "Preset"], horizontal=True, key="p_src")
    presets = {"2024 replay (R+1.5)": -1.5, "2020 replay (D+4.5)": 4.5, "Dead heat": 0.0,
               "Democratic wave (D+7)": 7.0, "Republican wave (R+6)": -6.0,
               f"Current midterm mood ({fmt_m(gen_avg or 0)})": round(gen_avg or 0, 1)}
    with c2:
        if src == "Preset":
            pick = st.selectbox("Preset", list(presets), key="p_preset")
            ss.nat_pres = presets[pick]
        elif src == "Poll average":
            if pres_avg is None:
                st.warning("Add 2028 polls on the Polls tab first.")
            else:
                ss.nat_pres = round(pres_avg, 1)
                st.caption(f"Weighted average of {len(ss.pres_polls)} polls: **{fmt_m(pres_avg)}**")
        nat = st.slider("National popular vote margin (D − R)", -15.0, 15.0, float(ss.nat_pres), 0.1,
                        disabled=src != "Manual", key=f"nat_slider_{src}_{ss.nat_pres}")
        if src == "Manual":
            ss.nat_pres = nat
    with c3:
        sig_default = model.default_uncertainty(DAYS_PRES)
        sigma_nat_p = st.slider("National error (pts SD)", 0.5, 8.0, round(sig_default, 1), 0.1,
                                help=f"Auto-set from {DAYS_PRES} days to election day. Lower it to see an election-eve view.")

    adj_t = tuple(sorted(ss.adj_pres.items()))
    lock_t = tuple(sorted(ss.locks_pres.items()))
    with st.spinner("Running simulations…"):
        res = run_pres(float(ss.nat_pres), sigma_nat_p, sigma_reg, sigma_state, elastic, adj_t, lock_t, n_sims)

    u = res.units
    med = int(np.median(res.ev_d))
    lo, hi = np.percentile(res.ev_d, [10, 90]).astype(int)
    big_numbers("Democrat wins", pct(res.p_d), f"median {med} EV · 80% range {lo}–{hi}",
                f"National {fmt_m(ss.nat_pres)}<br>tie 269–269: {pct(res.p_tie)}",
                "Republican wins", pct(res.p_r), f"median {538-med} EV · 80% range {538-hi}–{538-lo}")
    st.plotly_chart(bar_270(u), width="stretch", config={"displayModeBar": False})

    mcol, ecol = st.columns([2.2, 1])
    states_df = u[u["unit"] == u["state"]].copy()
    hover = [f"<b>{r.name}</b> · {int(r.ev)} EV{' statewide (+ districts below)' if r.unit in ('ME', 'NE') else ''}"
             f"<br>Projected {fmt_m(r.mean_margin)} (80%: {fmt_m(r.lo)} to {fmt_m(r.hi)})"
             f"<br>P(Dem) {r.p_d:.0%} · {model.rating(r.p_d)}"
             f"<br>2024: {fmt_m(r.base)}{'<br>🔒 Locked ' + r.lock if r.lock else ''}"
             for r in states_df.itertuples()]
    with mcol:
        st.markdown("##### Win probability by state  <span class='ss-note'>· click a state to edit it</span>",
                    unsafe_allow_html=True)
        ev_map = st.plotly_chart(us_map(states_df, "unit", "p_d", hover), width="stretch",
                                 on_select="rerun", selection_mode="points", key="pres_map",
                                 config={"displayModeBar": False, "scrollZoom": False})
        picked = selected_from_event(ev_map, set(ed.STATES))
        if picked and picked != ss.get("_last_pick_p"):
            ss.sel_state = picked
            ss["_last_pick_p"] = picked
        d_cols = st.columns(5)
        for col, dist in zip(d_cols, ed.DISTRICTS):
            row = u[u["unit"] == dist].iloc[0]
            col.metric(dist, pct(row.p_d) + " D", fmt_m(row.mean_margin), delta_color="off")

    with ecol:
        st.markdown("##### Scenario editor")
        options = list(ed.STATES) + list(ed.DISTRICTS)
        sel = st.selectbox("State or district", options, index=options.index(ss.sel_state),
                           format_func=lambda k: f"{k} · {(ed.STATES.get(k) or ed.DISTRICTS[k])[0]}")
        ss.sel_state = sel
        row = u[u["unit"] == sel].iloc[0]
        rc = model.rating(row.p_d)
        st.markdown(
            f"<span class='ss-chip' style='background:{RATING_COLORS[rc]};color:{'white' if 'Safe' in rc or 'Likely' in rc else INK}'>{rc}</span> "
            f"&nbsp;<b>{row.ev} EV</b> · P(D) {row.p_d:.0%}", unsafe_allow_html=True)
        st.caption(f"2024: {fmt_m(row.base)} · elasticity {row.elasticity:.2f} · region: {row.region}")
        a = st.slider("Local adjustment vs national swing (pts, + = D)", -15.0, 15.0,
                      float(ss.adj_pres.get(sel, 0.0)), 0.5, key=f"adj_{sel}")
        lk = st.radio("Call it", ["Model", "Force D", "Force R"], horizontal=True,
                      index=["", "D", "R"].index(ss.locks_pres.get(sel, "")), key=f"lock_{sel}")
        new_lock = {"Model": "", "Force D": "D", "Force R": "R"}[lk]
        changed = False
        if a != ss.adj_pres.get(sel, 0.0):
            ss.adj_pres[sel] = a; changed = True
        if new_lock != ss.locks_pres.get(sel, ""):
            if new_lock: ss.locks_pres[sel] = new_lock
            else: ss.locks_pres.pop(sel, None)
            changed = True
        if changed:
            st.rerun()
        active = {k: v for k, v in ss.adj_pres.items() if v}
        if active or ss.locks_pres:
            st.caption("Active edits: " + ", ".join(
                [f"{k} {v:+.1f}" for k, v in active.items()] + [f"{k}→{v}" for k, v in ss.locks_pres.items()]))
        st.metric("Tipping-point chance", f"{row.tipping:.1%}",
                  help="Share of simulations in which this unit casts the decisive 270th electoral vote.")
        st.markdown("<div class='ss-note'>Tip: Force a few states and watch the odds re-balance across the "
                    "correlated region. A Sun Belt shock moves AZ, NV, GA and NC together.</div>",
                    unsafe_allow_html=True)

    st.plotly_chart(snake(u), width="stretch", config={"displayModeBar": False})
    h1, h2 = st.columns([1.6, 1])
    h1.plotly_chart(ev_hist(res.ev_d), width="stretch", config={"displayModeBar": False})
    names = {k: v[0] for k, v in ed.STATES.items()} | {k: v[0] for k, v in ed.DISTRICTS.items()}
    h2.plotly_chart(tipping_chart(res.tipping, names, "Tipping-point states"), width="stretch",
                    config={"displayModeBar": False})

    with st.expander("Full state table", expanded=False):
        t = u.assign(Rating=u["p_d"].apply(model.rating)).sort_values("mean_margin")
        st.dataframe(
            t[["unit", "name", "ev", "base", "mean_margin", "lo", "hi", "p_d", "tipping", "Rating", "region", "elasticity"]]
            .rename(columns={"unit": "Unit", "name": "Name", "ev": "EV", "base": "2024", "mean_margin": "Projected",
                             "lo": "10th pct", "hi": "90th pct", "p_d": "P(D)", "tipping": "Tipping",
                             "region": "Region", "elasticity": "Elasticity"}),
            width="stretch", hide_index=True,
            column_config={"P(D)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                           "Tipping": st.column_config.NumberColumn(format="%.3f"),
                           "2024": st.column_config.NumberColumn(format="%+.1f"),
                           "Projected": st.column_config.NumberColumn(format="%+.1f"),
                           "10th pct": st.column_config.NumberColumn(format="%+.1f"),
                           "90th pct": st.column_config.NumberColumn(format="%+.1f")})

# ---------------------------------------------------------------- MIDTERMS
with tab_m:
    c1, c2, c3 = st.columns([1.3, 1.3, 1])
    with c1:
        gsrc = st.radio("Generic ballot from", ["Poll average", ed.AGGREGATOR_ANCHOR["label"], "Manual"],
                        horizontal=False, key="g_src")
    with c2:
        if gsrc == "Poll average":
            g_val = round(gen_avg if gen_avg is not None else 0.0, 1)
            st.caption(f"Weighted average of {len(ss.generic_polls)} polls (effective n ≈ {gen_eff:.1f}). "
                       "Edit polls on the Polls tab.")
        elif gsrc == "Manual":
            g_val = None
        else:
            g_val = ed.AGGREGATOR_ANCHOR["margin"]
        generic = st.slider("Generic congressional ballot (D − R)", -15.0, 15.0,
                            float(g_val if g_val is not None else ss.get("g_manual", 5.0)), 0.1,
                            disabled=g_val is not None, key=f"gen_{gsrc}_{g_val}")
        if g_val is None:
            ss.g_manual = generic
    with c3:
        sigma_nat_m = st.slider("Generic-ballot error (pts SD)", 0.5, 8.0,
                                round(model.default_uncertainty(DAYS_MID, base=2.5), 1), 0.1,
                                help="Historical midterm generic-ballot polling misses run ~2–4 pts.")

    races = ss.senate_races
    sen = run_senate(races.to_json(), generic, sigma_nat_m, sigma_reg, sigma_state,
                     tuple(sorted(ss.locks_sen.items())), n_sims)
    house_seats, p_house = run_house(generic, sigma_nat_m, swing_ratio, map_bias, n_sims)

    s_med = int(np.median(sen.seats_d))
    h_med = int(np.median(house_seats))
    st.markdown("#### Senate")
    big_numbers("Democratic control", pct(sen.p_d_control), f"median {s_med} seats · need 51",
                f"Generic {fmt_m(generic)}<br>50–50 goes R (VP tiebreak)",
                "Republican control", pct(1 - sen.p_d_control), f"median {100 - s_med} seats · need 50")

    r = sen.races
    sm, se = st.columns([2.2, 1])
    with sm:
        hov = [f"<b>{row['Name']}</b> · held {row['Held by']} ({row['Status']})"
               f"<br>Projected {fmt_m(row['Mean margin'])} · P(Dem) {row['P(D)']:.0%}"
               f"<br>{model.rating(row['P(D)'])}" + (f"<br><i>{row['Note']}</i>" if row['Note'] else "")
               for _, row in r.iterrows()]
        no_race = [s for s in ed.STATES if s not in ed.SENATE_2026 and s != "DC"]
        st.markdown("##### 35 Senate races  <span class='ss-note'>· grey = no race in 2026 · click to edit</span>",
                    unsafe_allow_html=True)
        sev = st.plotly_chart(us_map(r, "State", "P(D)", hov, missing=no_race), width="stretch",
                              on_select="rerun", selection_mode="points", key="sen_map",
                              config={"displayModeBar": False})
        pk = selected_from_event(sev, set(ed.SENATE_2026))
        if pk and pk != ss.get("_last_pick_s"):
            ss["sen_sel"] = pk
            ss["_last_pick_s"] = pk
    with se:
        st.markdown("##### Race editor")
        opts = list(ed.SENATE_2026)
        cur = ss.get("sen_sel", "ME")
        sel_s = st.selectbox("Race", opts, index=opts.index(cur), format_func=lambda k: f"{k} · {ed.STATES[k][0]}")
        ss["sen_sel"] = sel_s
        row = r[r["State"] == sel_s].iloc[0]
        rc = model.rating(row["P(D)"])
        st.markdown(f"<span class='ss-chip' style='background:{RATING_COLORS[rc]};color:{'white' if 'Safe' in rc or 'Likely' in rc else INK}'>{rc}</span>"
                    f" &nbsp;P(D) {row['P(D)']:.0%} · {fmt_m(row['Mean margin'])}", unsafe_allow_html=True)
        st.caption(f"Held by {row['Held by']} · {row['Status']} · 2024 pres {fmt_m(row['2024 pres margin'])}"
                   + (f" · {row['Note']}" if row["Note"] else ""))
        i = races.index[races["State"] == sel_s][0]
        ca = st.slider("Candidate effect (pts, + = D)", -20.0, 20.0, float(races.at[i, "Candidate adj"]), 0.5,
                       key=f"ca_{sel_s}")
        lk = st.radio("Call it", ["Model", "Force D", "Force R"], horizontal=True,
                      index=["", "D", "R"].index(ss.locks_sen.get(sel_s, "")), key=f"slock_{sel_s}")
        nl = {"Model": "", "Force D": "D", "Force R": "R"}[lk]
        ch = False
        if ca != races.at[i, "Candidate adj"]:
            ss.senate_races.at[i, "Candidate adj"] = ca; ch = True
        if nl != ss.locks_sen.get(sel_s, ""):
            if nl: ss.locks_sen[sel_s] = nl
            else: ss.locks_sen.pop(sel_s, None)
            ch = True
        if ch:
            st.rerun()
        st.metric("Tipping-point chance", f"{row['Tipping %']:.1f}%")

    a1, a2 = st.columns(2)
    vals, cnt = np.unique(sen.seats_d, return_counts=True)
    fig = go.Figure(go.Bar(x=vals, y=cnt / cnt.sum(), marker_color=[D_BLUE if v >= 51 else R_RED for v in vals],
                           hovertemplate="%{x} D seats: %{y:.1%}<extra></extra>"))
    fig.add_vline(x=50.5, line_dash="dot", line_color=INK)
    base_layout(fig, h=300, title=dict(text="Democratic Senate seats", font=dict(size=14)))
    fig.update_yaxes(tickformat=".0%")
    a1.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    a2.plotly_chart(tipping_chart(sen.tipping, {k: v[0] for k, v in ed.STATES.items()}, "Tipping-point Senate races"),
                    width="stretch", config={"displayModeBar": False})

    st.markdown("#### House")
    big_numbers("Democratic majority", pct(p_house), f"median {h_med} seats · need 218",
                f"Break-even at {fmt_m(map_bias)}<br>{swing_ratio:g} seats / pt",
                "Republican majority", pct(1 - p_house), f"median {435 - h_med} seats")
    b1, b2 = st.columns(2)
    vals, cnt = np.unique(house_seats, return_counts=True)
    fig = go.Figure(go.Bar(x=vals, y=cnt / cnt.sum(), marker_color=[D_BLUE if v >= 218 else R_RED for v in vals],
                           marker_line_width=0, hovertemplate="%{x} D seats: %{y:.2%}<extra></extra>"))
    fig.add_vline(x=217.5, line_dash="dot", line_color=INK)
    base_layout(fig, h=300, title=dict(text="Democratic House seats", font=dict(size=14)))
    fig.update_yaxes(tickformat=".1%")
    b1.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    gx = np.linspace(-12, 15, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gx, y=model.house_curve(gx, swing_ratio, map_bias), mode="lines",
                             line=dict(color="#6E5BA8", width=3), hovertemplate="Generic %{x:.1f} → %{y:.0f} D seats<extra></extra>"))
    fig.add_hline(y=217.5, line_dash="dot", line_color=INK)
    fig.add_trace(go.Scatter(x=[generic], y=[model.house_curve(generic, swing_ratio, map_bias)], mode="markers+text",
                             marker=dict(size=13, color=D_BLUE if generic > map_bias else R_RED, line=dict(color="white", width=2)),
                             text=[f"  {fmt_m(generic)}"], textposition="middle right"))
    fig.add_trace(go.Scatter(x=[ed.NATIONAL_2024_HOUSE], y=[215], mode="markers+text",
                             marker=dict(size=9, color=MUTED, symbol="diamond"), text=["2024: 215 D  "],
                             textposition="middle left"))
    base_layout(fig, h=300, title=dict(text="Seats–votes curve", font=dict(size=14)))
    fig.update_xaxes(title="Generic ballot (D − R)")
    fig.update_yaxes(title="Expected D seats", range=[120, 320])
    b2.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with st.expander("Edit every Senate race (candidate effects, elasticity)"):
        edited = st.data_editor(
            ss.senate_races, hide_index=True, width="stretch", key="race_editor",
            disabled=["State", "Name", "Held by", "Status", "2024 pres margin", "Region"],
            column_config={"Candidate adj": st.column_config.NumberColumn(min_value=-25, max_value=25, step=0.5, format="%+.1f"),
                           "Elasticity": st.column_config.NumberColumn(min_value=0.3, max_value=1.6, step=0.05)})
        if not edited.equals(ss.senate_races):
            ss.senate_races = edited
            st.rerun()
        st.caption("Candidate effects are model assumptions, loosely calibrated to public race ratings "
                   "(early Oct 2026). Incumbents get +3 pts, appointed incumbents +1.5.")

# ------------------------------------------------------------------- POLLS
with tab_polls:
    p1, p2 = st.columns(2)
    for col, key, title, avg, eff, table in [
        (p1, "generic_polls", "Generic congressional ballot", gen_avg, gen_eff, gen_table),
        (p2, "pres_polls", "2028 presidential (hypothetical matchups)", pres_avg, pres_eff, pres_table),
    ]:
        with col:
            st.markdown(f"#### {title}")
            st.metric("Weighted average", fmt_m(avg) if avg is not None else "—",
                      help=f"Effective number of polls: {eff:.1f}")
            new = st.data_editor(
                ss[key], num_rows="dynamic", width="stretch", hide_index=True, key=f"ed_{key}",
                column_config={
                    "End date": st.column_config.TextColumn(help="YYYY-MM-DD"),
                    "Margin (D-R)": st.column_config.NumberColumn(step=0.5, format="%+.1f"),
                    "Sample": st.column_config.NumberColumn(min_value=50, step=50),
                    "Population": st.column_config.SelectboxColumn(options=["LV", "RV", "A"]),
                })
            if not new.equals(ss[key]):
                ss[key] = new
                st.rerun()
            if table is not None and len(table):
                fig = go.Figure(go.Bar(x=table["Weight %"], y=table["Pollster"], orientation="h",
                                       marker_color=["#2E5EAA" if m > 0 else "#C23B33" for m in table["Margin (D-R)"]],
                                       text=[fmt_m(m) for m in table["Margin (D-R)"]], textposition="outside",
                                       hovertemplate="%{y}: %{x:.1f}% of weight<extra></extra>"))
                base_layout(fig, h=60 + 34 * len(table), title=dict(text="Share of the average", font=dict(size=13)))
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(ticksuffix="%", range=[0, table["Weight %"].max() * 1.3])
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.caption("Seeded generic-ballot polls come from public reporting in early Oct 2026; summer poll dates are "
               "approximate and sample sizes default to 1,000 where not reported. 2028 polls are placeholders: "
               "replace them with real matchup polls once they exist.")

    st.markdown("#### How the model works")
    st.markdown(f"""
1. **Polls → environment.** Each poll gets weight `√(n/1000) × 0.5^(age / {half_life} days) × population` (LV 1.0, RV 0.9, adults 0.7).
2. **Environment → states.** Each state starts from its 2024 result and moves by *elasticity × (national − 2024 national)*, plus any edit you make.
3. **Correlated uncertainty.** Every simulated election draws one fat-tailed national error (Student-t, 5 df), one error per region (Rust Belt, Sun Belt, Northeast, South, Plains, Mountain, Pacific), and independent state noise. States that share a region miss together.
4. **Count.** {n_sims:,} elections are tallied for win probabilities, EV distributions and the tipping point: the state that delivers the 270th vote.
5. **Senate** uses the same engine with the generic ballot measured against the 2024 House vote ({fmt_m(ed.NATIONAL_2024_HOUSE)}), plus incumbency and candidate effects. **House** uses a seats–votes curve with adjustable swing ratio and map bias.

*Limits:* this is a scenario tool, not a forecast. Baselines are 2024 results, candidate effects are assumptions, and no fundamentals model (economy, approval) is included.
""")
