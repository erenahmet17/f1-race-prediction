"""F1 tahmin projesinin yerel Streamlit arayüzü."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_prediction.collection import WeekendDataPaths, cache_weekend_laps  # noqa: E402
from f1_prediction.dashboard import (  # noqa: E402
    available_rounds,
    available_seasons,
    build_race_view,
    load_backtest,
    load_history_round,
    load_model_summary,
    load_saved_prediction,
    load_saved_retrospective,
    style_team_rows,
    turkish_upper,
)
from f1_prediction.prediction import (  # noqa: E402
    create_pre_race_prediction,
    create_retrospective_prediction,
)


st.set_page_config(
    page_title="PIT WALL | F1 Prediction",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1250px; padding-top: 3.75rem; }
    .pit-hero {
        background: linear-gradient(135deg, #1c2432 0%, #141922 65%, #26171d 100%);
        border: 1px solid #3b4658;
        border-left: 8px solid #f34b50;
        box-shadow: 10px 10px 0 #0b0e14;
        padding: 2.2rem 1.8rem 1rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .pit-kicker {
        color: #ffd166;
        font: 700 .77rem Consolas, monospace;
        letter-spacing: .20em;
    }
    .pit-title {
        color: #f6f6f4;
        font: 900 clamp(2.2rem, 5vw, 4.5rem) Consolas, monospace;
        letter-spacing: -.06em;
        line-height: 1.04;
        margin: .7rem 0 .45rem;
        position: relative;
        z-index: 2;
        text-shadow: 0 2px 10px #141922;
    }
    .pit-title span { color: #f34b50; }
    .pit-subtitle { color: #abb5c6; font-size: 1rem; max-width: 40rem; }
    .pit-pixels {
        position: absolute; right: 1.7rem; top: 1.5rem;
        opacity: .95; width: min(27%, 220px);
        z-index: 1;
    }
    .pit-stripe {
        background: repeating-linear-gradient(90deg, #f6f6f4 0 16px, #141922 16px 32px);
        height: 7px; margin-top: 1.4rem; opacity: .75;
    }
    .pit-tag {
        display: inline-block; border: 1px solid #f34b50;
        color: #ffd166; padding: .26rem .5rem; margin: .4rem 0 .9rem;
        font: 700 .72rem Consolas, monospace; letter-spacing: .08em;
    }
    .pit-section {
        color: #ffd166; font: 700 .82rem Consolas, monospace;
        letter-spacing: .16em; margin-top: 1rem;
    }
    /* Sekme görünür olduğunda panel bir saniye beklerken araba pisti geçer. */
    [role="tabpanel"]:not([hidden]) {
        position: relative;
        min-height: 70px;
        overflow-x: clip;
    }
    [role="tabpanel"]:not([hidden]) > [data-testid="stVerticalBlock"] {
        animation: pit-panel-enter .25s ease-out 1s both;
    }
    [role="tabpanel"]:not([hidden])::before {
        content: "";
        position: absolute;
        top: 3px;
        left: -130px;
        width: 120px;
        height: 56px;
        z-index: 3;
        pointer-events: none;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 240 112' shape-rendering='crispEdges'%3E%3Cpath fill='%23f34b50' d='M60 17h95v11H60zm21 11h55v10H81zM45 48h145v12H45zM17 60h204v15H17z'/%3E%3Cpath fill='%23b83040' d='M4 75h220v12H4z'/%3E%3Cpath fill='%23ffd166' d='M96 38h34v10H96zM9 49h34v10H9zm189 3h27v10h-27z'/%3E%3Cpath fill='%230b0e14' d='M23 76h37v24H23zm139 0h37v24h-37z'/%3E%3Cpath fill='%234a5567' d='M30 82h23v18H30zm139 0h23v18h-23z'/%3E%3C/svg%3E") center / contain no-repeat;
        animation: pit-car-pass 1s linear both;
    }
    @keyframes pit-car-pass {
        0% { left: -130px; opacity: 1; }
        85% { opacity: 1; }
        100% { left: calc(100% + 10px); opacity: 0; }
    }
    @keyframes pit-panel-enter {
        from { opacity: 0; transform: translateY(7px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 720px) {
        .block-container { padding-top: 3.2rem; }
        .pit-hero { padding-top: 2rem; }
        .pit-pixels { opacity: .22; width: 115px; top: auto; bottom: .45rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        [role="tabpanel"]:not([hidden]) > [data-testid="stVerticalBlock"] { animation: none; }
        [role="tabpanel"]:not([hidden])::before { display: none; }
    }
    </style>
    <div class="pit-hero">
      <div class="pit-kicker">// RACE STRATEGY TERMINAL · V2</div>
      <div class="pit-title">PIT<span>WALL</span></div>
      <div class="pit-subtitle">Yarış tahminleri, gerçek sonuçlar ve modelin tur tur performansı.</div>
      <svg class="pit-pixels" viewBox="0 0 240 112" shape-rendering="crispEdges" aria-hidden="true">
        <rect x="60" y="17" width="95" height="11" fill="#f34b50"/>
        <rect x="81" y="28" width="55" height="10" fill="#f34b50"/>
        <rect x="96" y="38" width="34" height="10" fill="#ffd166"/>
        <rect x="45" y="48" width="145" height="12" fill="#f34b50"/>
        <rect x="17" y="60" width="204" height="15" fill="#f34b50"/>
        <rect x="4" y="75" width="220" height="12" fill="#b83040"/>
        <rect x="23" y="76" width="37" height="24" fill="#0b0e14"/>
        <rect x="30" y="82" width="23" height="18" fill="#4a5567"/>
        <rect x="162" y="76" width="37" height="24" fill="#0b0e14"/>
        <rect x="169" y="82" width="23" height="18" fill="#4a5567"/>
        <rect x="9" y="49" width="34" height="10" fill="#ffd166"/>
        <rect x="198" y="52" width="27" height="10" fill="#ffd166"/>
      </svg>
      <div class="pit-stripe"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


paths = WeekendDataPaths.from_project_root(PROJECT_ROOT)
processed = paths.processed
seasons = available_seasons(processed)
if not seasons:
    st.info("Henüz işlenmiş sezon verisi yok. Önce yarış verilerini hazırlayın.")
    st.stop()

with st.sidebar:
    st.markdown("### 🏁 PIT WALL")
    season = st.selectbox("Sezon", seasons, key="season")
    st.caption("Veriler projenin data/processed klasöründen okunur.")
    st.divider()
    st.markdown("**Model:** PenaltyAwareHybrid")
    st.caption("Grid cezası olan pilotlar için özel sıralama yaklaşımı kullanılır.")

backtest, round_metrics = load_backtest(processed, season)
summary = load_model_summary(processed, season)
rounds = available_rounds(processed, season, backtest)

overview_tab, races_tab, predict_tab, retrospective_tab = st.tabs(
    ["🏁 Genel bakış", "📋 Yarışlar", "🚀 Yeni tahmin", "⏪ Geriye dönük"]
)


def show_race_rows(rows: pd.DataFrame, *, height: int = 690) -> None:
    if rows.empty:
        return
    display = rows.drop(columns=["DriverId", "CleanMAEEligible"], errors="ignore")
    display = display.rename(columns={
        "PredictedPosition": "Tahmin",
        "Driver": "Pilot",
        "Team": "Takım",
        "StartOrder": "Grid",
        "ActualPosition": "Sonuç",
        "AbsoluteRankError": "Hata",
        "RaceStatusCategory": "Durum",
    })
    display = display[[
        column for column in
        ("Tahmin", "Pilot", "Takım", "Grid", "Sonuç", "Hata", "Durum")
        if column in display
    ]]
    for column in ("Sonuç", "Hata"):
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").astype("Int64")
    st.dataframe(
        style_team_rows(display, team_column="Takım"),
        hide_index=True, width="stretch", height=height,
    )

with overview_tab:
    st.markdown('<div class="pit-section">01 / MODEL DURUMU</div>', unsafe_allow_html=True)
    st.subheader(f"{season} sezonu")
    if summary.empty:
        st.info("Bu sezon için backtest sonucu henüz yok.")
    else:
        hybrid = summary.loc[summary["Model"].eq("PenaltyAwareHybrid")]
        grid = summary.loc[summary["Model"].eq("Grid")]
        if not hybrid.empty:
            current = hybrid.iloc[0]
            grid_mae = float(grid.iloc[0]["OverallMAE"]) if not grid.empty else None
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sıralama MAE", f"{float(current['OverallMAE']):.2f}")
            c2.metric("CleanMAE", f"{float(current['OverallCleanMAE']):.2f}")
            c3.metric("Test edilen yarış", int(current["RoundsEvaluated"]))
            c4.metric(
                "Grid'e göre fark",
                f"{float(current['OverallMAE']) - grid_mae:+.2f}" if grid_mae is not None else "—",
            )
        st.caption(
            "Bu ölçümler geçmiş yarışlar üzerinde, her yarış için yalnız önceki turlarla "
            "yapılan geriye dönük testten gelir. Canlı tahmin kaydı değildir."
        )

        st.markdown('<div class="pit-section">02 / MODEL KARŞILAŞTIRMASI</div>', unsafe_allow_html=True)
        comparison = summary[["Model", "OverallMAE", "OverallCleanMAE"]].rename(
            columns={"Model": "Model", "OverallMAE": "MAE", "OverallCleanMAE": "CleanMAE"}
        )
        st.dataframe(comparison.round(2), hide_index=True, width="stretch")

        if not round_metrics.empty:
            metrics = round_metrics.loc[
                round_metrics["Model"].isin(["PenaltyAwareHybrid", "Grid"]),
                ["RoundNumber", "Model", "MAE", "CleanMAE"],
            ]
            if not metrics.empty:
                values = metrics.pivot(
                    index="RoundNumber", columns="Model", values=["MAE", "CleanMAE"]
                )
                series = (
                    ("MAE", "PenaltyAwareHybrid", "Hibrit · MAE", "#f34b50"),
                    ("CleanMAE", "PenaltyAwareHybrid", "Hibrit · CleanMAE", "#ff9aa2"),
                    ("MAE", "Grid", "Grid · MAE", "#ffd166"),
                    ("CleanMAE", "Grid", "Grid · CleanMAE", "#ffe6a0"),
                )
                chart = pd.DataFrame(index=values.index)
                colors = []
                for metric, model, label, color in series:
                    if (metric, model) in values:
                        chart[label] = values[(metric, model)]
                        colors.append(color)
                st.markdown('<div class="pit-section">03 / TUR TUR HATA</div>', unsafe_allow_html=True)
                st.line_chart(chart, color=colors, x_label="Round", y_label="Sıra hatası")

with races_tab:
    st.markdown('<div class="pit-section">YARIŞ ARŞİVİ</div>', unsafe_allow_html=True)
    if not rounds:
        st.info("Bu sezon için kayıtlı yarış bulunamadı.")
    else:
        selected_round = st.selectbox(
            "Yarış turu", rounds, index=len(rounds) - 1,
            format_func=lambda number: f"ROUND {number:02d}",
            key="race_round",
        )
        history = load_history_round(processed, season, selected_round)
        saved = load_saved_prediction(processed, season, selected_round)
        retrospective = load_saved_retrospective(processed, season, selected_round)
        round_backtest = (
            backtest.loc[backtest["RoundNumber"].eq(selected_round)]
            if not backtest.empty else pd.DataFrame()
        )
        race = build_race_view(
            saved_prediction=saved,
            retrospective_prediction=retrospective,
            backtest_prediction=round_backtest,
            history=history,
        )
        st.subheader(race.event_name)
        if race.source:
            st.markdown(f'<div class="pit-tag">{turkish_upper(race.source)}</div>', unsafe_allow_html=True)
        else:
            st.caption("Bu tur için tahmin kaydı yok; yalnız resmî yarış sonucu gösteriliyor.")

        if race.mae is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Yarış MAE", f"{race.mae:.2f}")
            c2.metric("CleanMAE", f"{race.clean_mae:.2f}" if race.clean_mae is not None else "—")
            c3.metric("Pilot", len(race.rows))
        elif race.source:
            st.info("Yarış sonucu henüz kayıtlı değil. MAE, sonuç eklendiğinde görünecek.")

        show_race_rows(race.rows)

with predict_tab:
    st.markdown('<div class="pit-section">GELECEK YARIŞ</div>', unsafe_allow_html=True)
    st.subheader("Yeni yarış tahmini")
    completed = [number for number in rounds if not load_history_round(processed, season, number).empty]
    next_round = max(completed, default=1) + 1
    st.caption(
        "Practice, sprint, ana sıralama ve FIA final grid açıklandıktan sonra çalıştırın. "
        "Yeni kayıt yalnız yarış başlamadan önce oluşturulur. OpenF1 canlı seans sırasında "
        "kapalı olduğu için turları sıralamadan sonra önceden indirin."
    )
    with st.form("prediction_form"):
        target_round = st.number_input(
            "Tahmin edilecek round", min_value=2, max_value=40,
            value=max(2, next_round), step=1,
        )
        submitted = st.form_submit_button("🏎️ Tahmini oluştur", type="primary")
        cache_submitted = st.form_submit_button("💾 Turları önceden indir")
    if cache_submitted:
        with st.spinner("Antrenman ve sprint turları indiriliyor..."):
            try:
                lap_counts = cache_weekend_laps(season, int(target_round), paths=paths)
            except Exception as exc:
                st.error(f"Turlar indirilemedi: {exc}")
            else:
                lap_summary = " · ".join(
                    f"{name}: {count} tur" for name, count in lap_counts.items()
                )
                if all(lap_counts.values()):
                    st.success(lap_summary)
                else:
                    st.warning(
                        f"{lap_summary}. Eksik seanslar henüz bitmemiş veya OpenF1 canlı "
                        "seans nedeniyle kapalı olabilir; sonra yeniden deneyin."
                    )
    if submitted:
        with st.spinner("Hafta sonu verileri yükleniyor ve model eğitiliyor..."):
            try:
                result = create_pre_race_prediction(
                    season,
                    int(target_round),
                    paths=paths,
                    save=True,
                )
            except Exception as exc:
                st.error(f"Tahmin üretilemedi: {exc}")
            else:
                st.success(
                    f"{result.weekend.event_name} tahmini kaydedildi: "
                    f"{result.prediction_path}"
                )
                missing_practice = int(result.weekend.features["PracticeDataMissing"].sum())
                if missing_practice:
                    st.warning(
                        f"{missing_practice} pilot için antrenman verisi alınamadı; "
                        "tahmin bu veri olmadan üretildi."
                    )

    saved_rounds = [
        number for number in rounds
        if not load_saved_prediction(processed, season, number).empty
    ]
    if saved_rounds:
        latest_round = max(saved_rounds)
        latest = load_saved_prediction(processed, season, latest_round)
        st.markdown('<div class="pit-section">SON KAYDEDİLEN TAHMİN</div>', unsafe_allow_html=True)
        st.write(f"Round {latest_round:02d} · {latest['EventName'].iloc[0]}")
        columns = [
            column for column in ("PredictedPosition", "Driver", "Team", "StartOrder", "GridPenaltyReason")
            if column in latest
        ]
        st.dataframe(
            style_team_rows(latest[columns].sort_values("PredictedPosition")),
            hide_index=True,
            width="stretch",
            height=540,
        )
    else:
        st.info("Henüz kaydedilmiş canlı yarış öncesi tahmin yok.")

with retrospective_tab:
    st.markdown('<div class="pit-section">KAÇIRILAN YARIŞ</div>', unsafe_allow_html=True)
    st.subheader("Geriye dönük deneme")
    st.caption(
        "Yarış bittikten sonra eksik önceki yarışları indirir, yalnız hedef yarıştan "
        "önce bilinen verilerle tahmin yapar, ardından FIA sonucunu çekip MAE'yi "
        "hesaplar. Bu, yarıştan önce kaydedilmiş canlı tahmin değildir. "
        "FIA final sonucu yayımlanmadan çalışmaz."
    )
    with st.form("retrospective_form"):
        retrospective_round = st.number_input(
            "Geriye dönük denenecek round", min_value=2, max_value=40,
            value=max(2, max(rounds, default=1) + 1), step=1,
        )
        retrospective_submitted = st.form_submit_button(
            "⏪ Verileri çek, tahmin et ve değerlendir", type="primary"
        )
    if retrospective_submitted:
        with st.spinner("Önceki yarışlar, yarış hafta sonu ve resmî sonuç yükleniyor..."):
            try:
                retro_result = create_retrospective_prediction(
                    season, int(retrospective_round), paths=paths
                )
            except Exception as exc:
                st.error(f"Geriye dönük deneme tamamlanamadı: {exc}")
            else:
                st.success(
                    f"{retro_result.weekend.event_name} geriye dönük denemesi "
                    f"kaydedildi. MAE: {retro_result.mae:.2f} · "
                    f"CleanMAE: {retro_result.clean_mae:.2f}"
                    if retro_result.clean_mae is not None
                    else f"{retro_result.weekend.event_name} kaydedildi. MAE: {retro_result.mae:.2f}"
                )

    retro_directory = processed / "retrospectives" / str(season)
    retro_rounds = sorted(
        int(path.name.removeprefix("round_").removesuffix("_retrospective.csv"))
        for path in retro_directory.glob("round_*_retrospective.csv")
        if path.name.removeprefix("round_").removesuffix("_retrospective.csv").isdigit()
    ) if retro_directory.exists() else []
    if retro_rounds:
        shown_round = st.selectbox(
            "Kaydedilmiş geriye dönük deneme", retro_rounds,
            index=len(retro_rounds) - 1,
            format_func=lambda number: f"ROUND {number:02d}",
            key="retrospective_saved_round",
        )
        retro_prediction = load_saved_retrospective(processed, season, shown_round)
        actual = load_history_round(processed, season, shown_round)
        retro_view = build_race_view(
            saved_prediction=pd.DataFrame(),
            retrospective_prediction=retro_prediction,
            backtest_prediction=pd.DataFrame(),
            history=actual,
        )
        st.markdown(
            '<div class="pit-tag">YARIŞ SONRASI GERİYE DÖNÜK DENEME</div>',
            unsafe_allow_html=True,
        )
        st.subheader(retro_view.event_name)
        c1, c2 = st.columns(2)
        c1.metric("MAE", f"{retro_view.mae:.2f}" if retro_view.mae is not None else "—")
        c2.metric(
            "CleanMAE",
            f"{retro_view.clean_mae:.2f}" if retro_view.clean_mae is not None else "—",
        )
        show_race_rows(retro_view.rows)
