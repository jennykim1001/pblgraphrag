"""
dashboard.py — Streamlit 대시보드
교수자 중심 UI: 데이터 업로드 → 분석 실행 → 보고서 열람/다운로드

확정 지표 4개: relation_entity_ratio, cross_ratio, connectivity, community_cross_ratio
(coverage는 길이 의존적이므로 제외)

실행: streamlit run dashboard.py
"""
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

from config import *

import re as _re
def _md_bold_to_html(text):
    """마크다운 **볼드**를 HTML <b>로 변환"""
    return _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

# ── 한글 폰트 (Plotly용) ──
KOREAN_FONT = "Noto Sans KR"

# ── 확정 지표 (coverage 제외) ──
FINAL_METRICS = ["relation_entity_ratio", "cross_ratio", "connectivity", "community_cross_ratio"]
METRIC_LABELS = {
    "relation_entity_ratio": "관계-엔티티 비율",
    "cross_ratio": "에필로그→활용 교차 비율",
    "connectivity": "그래프 연결성",
    "community_cross_ratio": "범주 교차 비율",
}
METRIC_DESC = {
    "relation_entity_ratio": "요소 간 연결의 밀도 (높을수록 경험을 밀접하게 연결)",
    "cross_ratio": "에필로그→활용 전이 비율 (높을수록 회고와 적용이 구조적으로 연결)",
    "connectivity": "경험의 통합도 (높을수록 경험이 하나의 서사로 통합)",
    "community_cross_ratio": "범주 간 통합적 사고 (높을수록 다양한 주제를 연결)",
}

st.set_page_config(page_title="PBL 학습경험 평가 프레임워크", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    *{font-family:'Noto Sans KR',sans-serif}
    .main .block-container{padding-top:1.2rem;max-width:1200px}
    .mc{background:linear-gradient(135deg,#f8f9fa,#e9ecef);border-radius:12px;padding:16px;text-align:center;border-left:4px solid #4A90D9;margin-bottom:8px}
    .mc .v{font-size:1.8rem;font-weight:900;color:#1F4E79}.mc .l{font-size:.82rem;color:#666;margin-top:2px}
    .category-tag { display: inline-block; background: #e8f0fe; color: #1A3764; padding: 4px 12px; border-radius: 16px; margin: 2px 4px; font-size: 13px; }
    .transfer-path { background: #f0f7e8; border-left: 3px solid #4CAF50; padding: 8px 16px; margin: 6px 0; border-radius: 0 8px 8px 0; font-size: 13px; }
    .et{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.75rem;font-weight:500;margin:2px;color:white}
    div[data-testid="stSidebar"]{background:linear-gradient(180deg,#1F4E79,#2E75B6)}
    div[data-testid="stSidebar"] *{color:white!important}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 GraphRAG PBL\n### 동적 평가 프레임워크")
    st.markdown("---")
    page = st.radio("📌 메뉴", ["👤 개인별 보고서", "🏫 반별 보고서", "🎓 전공별 보고서"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📁 데이터")
    uploaded_files = st.file_uploader("반별 엑셀 파일 업로드 (.xlsx)", type=["xlsx"], accept_multiple_files=True, help="각 파일: 순번, 이름, 전공, 프로젝트주제, 에필로그, 활용 컬럼 필요")
    use_existing = st.checkbox("기존 추출 결과 사용", value=True, help="기존 JSON/캐시가 있으면 API 호출 없이 재사용")

def load_data_from_uploads(uploaded_files):
    frames = []
    for file in uploaded_files:
        df = pd.read_excel(file)
        fname = file.name.replace(".xlsx", "").replace("class", "")
        df["class_id"] = fname
        df["student_id"] = fname + "_" + df["이름"].astype(str)
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"전공": "major", "프로젝트주제": "project", "에필로그": "epilogue", "활용": "application"})
    result["epi_len"] = result["epilogue"].str.len()
    result["app_len"] = result["application"].str.len()
    result["total_len"] = result["epi_len"] + result["app_len"]
    return result

def load_data_from_disk():
    frames = []
    for class_id, filepath in CLASS_FILES.items():
        if filepath.exists():
            df = pd.read_excel(filepath)
            df["class_id"] = class_id
            df["student_id"] = class_id + "_" + df["이름"].astype(str)
            frames.append(df)
    if not frames: return None
    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"전공": "major", "프로젝트주제": "project", "에필로그": "epilogue", "활용": "application"})
    result["epi_len"] = result["epilogue"].str.len()
    result["app_len"] = result["application"].str.len()
    result["total_len"] = result["epi_len"] + result["app_len"]
    return result

def run_pipeline(df, use_existing_extractions=True):
    from pipeline import (step1_llm_extract, step1_load_existing, step2_merge_entities, step3_build_graph, step4_compute_metrics, step5_position_and_report)
    progress = st.progress(0, text="파이프라인 시작...")
    progress.progress(10, text="1단계: LLM 추출...")
    if use_existing_extractions:
        existing = step1_load_existing()
        if existing and len(existing) >= len(df): extractions = existing
        else: extractions = step1_llm_extract(df)
    else: extractions = step1_llm_extract(df)
    progress.progress(30, text="2단계: 통합적 엔티티 해상도...")
    merge_result = step2_merge_entities(extractions)
    extractions = merge_result["extractions"]
    progress.progress(50, text="3단계: 그래프 + 커뮤니티 탐지...")
    graph_results = step3_build_graph(df, extractions, merge_result)
    progress.progress(70, text="4단계: 구조적 지표 산출...")
    metrics_df = step4_compute_metrics(df, extractions, graph_results)
    progress.progress(90, text="5단계: 보고서 생성...")
    final_df = step5_position_and_report(df, metrics_df, extractions, graph_results)
    progress.progress(100, text="완료!")
    return {"df": df, "extractions": extractions, "graph_results": graph_results, "metrics_df": metrics_df, "final_df": final_df}

def render_overview_metrics(results):
    """메트릭 카드"""
    df = results["df"]; metrics = results["metrics_df"]
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="mc"><div class="v">{len(df)}</div><div class="l">분석 대상 (명)</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="mc"><div class="v">{df["class_id"].nunique()}</div><div class="l">분석 반 (개)</div></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="mc"><div class="v">{metrics["n_entities"].sum()}</div><div class="l">추출 엔티티 (총)</div></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="mc"><div class="v">{metrics["n_relations"].sum()}</div><div class="l">추출 관계 (총)</div></div>', unsafe_allow_html=True)

def render_positioning_map(results):
    """반별 포지셔닝 맵"""
    metrics = results["metrics_df"]

    # ── 반별 포지셔닝 맵 ──
    st.markdown("### 반별 포지셔닝 맵")

    # 종합의견 생성
    import plotly.graph_objects as go
    class_colors = {"A": "#4A90D9", "B": "#E8913A", "C": "#2ECC71", "D": "#E74C3C", "E": "#9B59B6"}
    class_ids = sorted(metrics["class_id"].unique())
    n_classes = len(class_ids)

    # 반별 핵심 지표 산출
    class_summaries = []
    best_transfer_class = None
    best_transfer_val = -1
    worst_transfer_class = None
    worst_transfer_val = 999
    total_zero = 0

    for cid in class_ids:
        cd = metrics[metrics["class_id"] == cid]
        cr_mean = cd["cross_ratio"].mean()
        cr_zero = len(cd[cd["cross_ratio"] == 0])
        rer_mean = cd["relation_entity_ratio"].mean()
        total_zero += cr_zero
        if cr_mean > best_transfer_val:
            best_transfer_val = cr_mean; best_transfer_class = cid
        if cr_mean < worst_transfer_val:
            worst_transfer_val = cr_mean; worst_transfer_class = cid
        class_summaries.append({"cid": cid, "n": len(cd), "rer": rer_mean, "cr": cr_mean, "zero": cr_zero})

    summary_lines = []
    summary_lines.append(f"전체 {len(metrics)}명 중 전이 부재 학생(교차 연결=0)은 **{total_zero}명**입니다.")
    summary_lines.append(f"전이 연결이 가장 활발한 반은 **반 {best_transfer_class}** (평균 {best_transfer_val:.3f}), "
                        f"가장 낮은 반은 **반 {worst_transfer_class}** (평균 {worst_transfer_val:.3f})입니다.")

    # 반 간 밀도 비교
    rer_values = [s["rer"] for s in class_summaries]
    if max(rer_values) - min(rer_values) < 0.05:
        summary_lines.append("구조적 밀도는 반 간 차이가 크지 않아, 동일 교과목의 성찰 구조가 유사하게 형성된 것으로 보입니다.")
    else:
        high_rer = max(class_summaries, key=lambda x: x["rer"])
        low_rer = min(class_summaries, key=lambda x: x["rer"])
        summary_lines.append(f"구조적 밀도는 반 {high_rer['cid']}({high_rer['rer']:.3f})이 가장 높고, 반 {low_rer['cid']}({low_rer['rer']:.3f})이 가장 낮습니다.")

    if total_zero > 0:
        summary_lines.append(f"⚠️ 전이 부재 학생 {total_zero}명에 대해 '학습 내용을 전공·일상에 어떻게 적용할 수 있는지' 성찰을 유도하는 수업 설계 보완이 필요합니다.")

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#EBF5FB,#D6EAF8);padding:14px 18px;border-radius:10px;margin-bottom:16px;font-size:.88rem;color:#1B4F72;border-left:4px solid #2E86C1">'
        f'<b>🕸️ 종합의견:</b> {_md_bold_to_html("  ".join(summary_lines))}</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:8px">X축: 관계-엔티티 비율 (구조적 밀도) / Y축: 교차 연결 비율 (에필로그→활용 전이). 버블 크기는 엔티티 수.</div>', unsafe_allow_html=True)

    # 전체 84명 기준 축 범위 통일
    x_global_min = max(0, metrics["relation_entity_ratio"].min() - 0.05)
    x_global_max = min(1.2, metrics["relation_entity_ratio"].max() + 0.05)
    y_global_min = max(-0.02, metrics["cross_ratio"].min() - 0.02)
    y_global_max = min(0.5, metrics["cross_ratio"].max() + 0.03)

    for i in range(0, n_classes, 2):
        pm_cols = st.columns(2)
        for j, pm_col in enumerate(pm_cols):
            idx = i + j
            if idx >= n_classes:
                break
            cid = class_ids[idx]
            cdata = metrics[metrics["class_id"] == cid]
            color = class_colors.get(cid, "#888")

            x_mid = cdata["relation_entity_ratio"].median()
            y_mid = cdata["cross_ratio"].median()

            with pm_col:
                fig = go.Figure()

                # 사분면 배경 (전체 범위 기준)
                fig.add_shape(type="rect", x0=x_global_min, y0=y_mid, x1=x_mid, y1=y_global_max,
                             fillcolor="rgba(46,204,113,0.06)", line=dict(width=0))
                fig.add_shape(type="rect", x0=x_mid, y0=y_mid, x1=x_global_max, y1=y_global_max,
                             fillcolor="rgba(46,204,113,0.14)", line=dict(width=0))
                fig.add_shape(type="rect", x0=x_global_min, y0=y_global_min, x1=x_mid, y1=y_mid,
                             fillcolor="rgba(231,76,60,0.06)", line=dict(width=0))
                fig.add_shape(type="rect", x0=x_mid, y0=y_global_min, x1=x_global_max, y1=y_mid,
                             fillcolor="rgba(241,196,15,0.08)", line=dict(width=0))

                fig.add_hline(y=y_mid, line_dash="dot", line_color="#CCC", line_width=1)
                fig.add_vline(x=x_mid, line_dash="dot", line_color="#CCC", line_width=1)

                # 사분면 라벨
                fig.add_annotation(x=x_global_min + 0.01, y=y_global_max - 0.005, text="밀도↓ 전이↑",
                                  showarrow=False, font=dict(size=8, color="#888"), xanchor="left", yanchor="top")
                fig.add_annotation(x=x_global_max - 0.01, y=y_global_max - 0.005, text="✦ 밀도↑ 전이↑",
                                  showarrow=False, font=dict(size=8, color="#1B5E20"), xanchor="right", yanchor="top")
                fig.add_annotation(x=x_global_min + 0.01, y=y_global_min + 0.005, text="밀도↓ 전이↓",
                                  showarrow=False, font=dict(size=8, color="#BF360C"), xanchor="left", yanchor="bottom")
                fig.add_annotation(x=x_global_max - 0.01, y=y_global_min + 0.005, text="밀도↑ 전이↓",
                                  showarrow=False, font=dict(size=8, color="#888"), xanchor="right", yanchor="bottom")

                # 학생 점
                names = [r["student_id"].split("_")[-1] for _, r in cdata.iterrows()]
                sizes = [max(20, r["n_entities"] * 3.5) for _, r in cdata.iterrows()]
                hovers = []
                for _, r in cdata.iterrows():
                    hovers.append(
                        f"<b>{r['student_id'].split('_')[-1]}</b> ({r['major']})<br>"
                        f"구조적 밀도: {r['relation_entity_ratio']:.3f}<br>"
                        f"교차 연결: {r['cross_ratio']:.3f}<br>"
                        f"엔티티: {r['n_entities']}개 / 관계: {r['n_relations']}개")

                fig.add_trace(go.Scatter(
                    x=cdata["relation_entity_ratio"], y=cdata["cross_ratio"],
                    mode='markers+text',
                    marker=dict(size=sizes, color=color,
                               line=dict(width=2, color='white'), opacity=0.85),
                    text=names, textposition="top center",
                    textfont=dict(size=9, color='#333'),
                    hovertext=hovers, hoverinfo='text',
                    name=f"반 {cid}"))

                fig.update_layout(
                    height=400, showlegend=False,
                    title=dict(text=f"반 {cid} ({len(cdata)}명)", font=dict(size=14)),
                    xaxis=dict(title="← 구조적 밀도 (관계-엔티티 비율) →",
                              range=[x_global_min, x_global_max], tickformat=".2f", gridcolor="#F0F0F0"),
                    yaxis=dict(title="← 전이 연결 (교차 연결 비율) →",
                              range=[y_global_min, y_global_max], tickformat=".3f", gridcolor="#F0F0F0"),
                    plot_bgcolor='white',
                    margin=dict(l=60, r=20, t=40, b=60))

                st.plotly_chart(fig, use_container_width=True, key=f"pm_{cid}")

                # 해석 텍스트
                rer_mean = cdata["relation_entity_ratio"].mean()
                cr_mean = cdata["cross_ratio"].mean()
                cr_std = cdata["cross_ratio"].std()
                cr_zero = len(cdata[cdata["cross_ratio"] == 0])
                q1 = len(cdata[(cdata["relation_entity_ratio"] >= x_mid) & (cdata["cross_ratio"] >= y_mid)])
                q2 = len(cdata[(cdata["relation_entity_ratio"] < x_mid) & (cdata["cross_ratio"] >= y_mid)])
                q3 = len(cdata[(cdata["relation_entity_ratio"] >= x_mid) & (cdata["cross_ratio"] < y_mid)])
                q4 = len(cdata[(cdata["relation_entity_ratio"] < x_mid) & (cdata["cross_ratio"] < y_mid)])

                # 사분면별 학생 이름
                q1_names = cdata[(cdata["relation_entity_ratio"] >= x_mid) & (cdata["cross_ratio"] >= y_mid)]["student_id"].apply(lambda x: x.split("_")[-1]).tolist()
                q3_names = cdata[(cdata["relation_entity_ratio"] >= x_mid) & (cdata["cross_ratio"] < y_mid)]["student_id"].apply(lambda x: x.split("_")[-1]).tolist()
                q4_names = cdata[(cdata["relation_entity_ratio"] < x_mid) & (cdata["cross_ratio"] < y_mid)]["student_id"].apply(lambda x: x.split("_")[-1]).tolist()
                zero_names = cdata[cdata["cross_ratio"] == 0]["student_id"].apply(lambda x: x.split("_")[-1]).tolist()

                # 구체적 해석 생성
                lines = []
                lines.append(f"**반 {cid} 해석**")
                lines.append(f"- 사분면 분포: ✦밀도↑전이↑ **{q1}명**, 밀도↓전이↑ {q2}명, 밀도↑전이↓ {q3}명, 밀도↓전이↓ {q4}명")

                if q1 > 0:
                    lines.append(f"- 🟢 **우수 학생**: {', '.join(q1_names[:5])} — 성찰 구조와 전이 연결 모두 양호. 좋은 성찰 사례로 공유 가능.")

                if cr_zero > 0:
                    lines.append(f"- 🔴 **전이 부재**: {', '.join(zero_names)} ({cr_zero}명) — 에필로그→활용 연결이 없음. '배운 것을 전공에 어떻게 적용할 수 있을지' 추가 질문 필요.")

                if q3 > 0:
                    lines.append(f"- 🟡 **전이 보완 필요**: {', '.join(q3_names[:5])} — 성찰은 풍부하나 전공·일상 적용으로 연결되지 않음. 전이 구상을 유도하는 피드백 필요.")

                if cr_std > 0.10:
                    lines.append(f"- ⚠️ 전이 편차가 큼 (SD={cr_std:.3f}). 학생 간 수준 차이가 크므로 개인별 맞춤 피드백이 효과적.")
                else:
                    lines.append(f"- 전이 수준이 비교적 균일 (SD={cr_std:.3f}).")

                st.markdown(
                    f'<div style="background:#F8F9FA;border-left:4px solid {color};padding:10px 14px;border-radius:0 8px 8px 0;margin:8px 0;font-size:.85rem;line-height:1.7">'
                    f'{_md_bold_to_html("<br>".join(lines))}</div>', unsafe_allow_html=True)

    # 사분면 해석 가이드
    st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px">
<div style="padding:10px;background:#E8F8F5;border-radius:8px;font-size:.85rem"><b>✦ 우상단: 구조적 밀도 높음 + 전이 활발</b><br>이상적 위치. 경험을 밀접히 연결하면서 전공·일상 적용도 구체적으로 구상.</div>
<div style="padding:10px;background:#FEF9E7;border-radius:8px;font-size:.85rem"><b>좌상단: 구조적 밀도 낮음 + 전이 활발</b><br>전이는 관찰되나, 경험 요소 간 연결을 더 풍부하게 형성하면 성찰이 심화될 수 있음.</div>
<div style="padding:10px;background:#FEF9E7;border-radius:8px;font-size:.85rem"><b>우하단: 구조적 밀도 높음 + 전이 부족</b><br>에필로그 내 연결은 잘 형성되나, 전공·일상 적용을 구체적으로 연결지어 볼 필요.</div>
<div style="padding:10px;background:#FDEDEC;border-radius:8px;font-size:.85rem"><b>좌하단: 구조적 밀도 낮음 + 전이 부족</b><br>경험 간 연결 형성 + 전공·일상 적용 구체화 모두 필요. 개별 면담 권장.</div>
</div>
""", unsafe_allow_html=True)

def render_category_compare(results):
    """반별 학습경험 상위 범주 비교"""
    metrics = results["metrics_df"]
    st.markdown("### 반별 학습경험 상위 범주 비교")
    st.caption("각 반 내 에필로그/활용 요소의 비율(%)로 표시합니다. 반마다 범주 이름과 구성이 다를 수 있습니다.")

    extractions = results["extractions"]
    from collections import Counter as _Ctr

    all_class_ids = sorted(metrics["class_id"].unique())
    n_classes = len(all_class_ids)

    for row_start in range(0, n_classes, 2):
        ov_cols = st.columns(2)
        for j, ov_col in enumerate(ov_cols):
            idx = row_start + j
            if idx >= n_classes:
                break
            cid = all_class_ids[idx]
            cd = metrics[metrics["class_id"] == cid]

            epi_cats = _Ctr()
            app_cats = _Ctr()
            for sid in cd["student_id"]:
                ext = extractions.get(sid, {"entities": []})
                for ent in ext.get("entities", []):
                    cat = ent.get("category", "기타")
                    if ent.get("source", "").startswith("에필로그"):
                        epi_cats[cat] += 1
                    elif ent.get("source", "").startswith("활용"):
                        app_cats[cat] += 1

            total_epi = sum(epi_cats.values())
            total_app = sum(app_cats.values())
            epi_pct = {c: (v / total_epi * 100 if total_epi > 0 else 0) for c, v in epi_cats.items()}
            app_pct = {c: (v / total_app * 100 if total_app > 0 else 0) for c, v in app_cats.items()}

            all_cats = sorted(set(list(epi_cats.keys()) + list(app_cats.keys())),
                            key=lambda x: -(epi_pct.get(x, 0) + app_pct.get(x, 0)))

            with ov_col:
                if all_cats and all_cats != ["기타"]:
                    import plotly.graph_objects as _go
                    fig_cat = _go.Figure()
                    fig_cat.add_trace(_go.Bar(
                        y=all_cats, x=[epi_pct.get(c, 0) for c in all_cats],
                        orientation='h', name='에필로그', marker_color='#4A90D9', opacity=0.85))
                    fig_cat.add_trace(_go.Bar(
                        y=all_cats, x=[app_pct.get(c, 0) for c in all_cats],
                        orientation='h', name='활용', marker_color='#E8913A', opacity=0.85))
                    fig_cat.update_layout(
                        barmode='group', height=max(250, len(all_cats) * 35),
                        title=dict(text=f"반 {cid} ({len(cd)}명)", font=dict(size=13)),
                        xaxis=dict(title="비율 (%)"),
                        yaxis=dict(autorange="reversed"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9)),
                        margin=dict(l=10, r=10, t=40, b=30), plot_bgcolor='white')
                    st.plotly_chart(fig_cat, use_container_width=True, key=f"ov_cat_{cid}")

                    top_epi = max(epi_pct.items(), key=lambda x: x[1]) if epi_pct else ("없음", 0)
                    top_app = max(app_pct.items(), key=lambda x: x[1]) if app_pct else ("없음", 0)
                    transfer_cats = [c for c in all_cats if app_pct.get(c, 0) > epi_pct.get(c, 0) and app_pct.get(c, 0) >= 15]

                    interp = f"**반 {cid}**: 에필로그 '{top_epi[0]}'({top_epi[1]:.0f}%) 중심 성찰. "
                    interp += f"활용 '{top_app[0]}'({top_app[1]:.0f}%) 중심 전이. "
                    if transfer_cats:
                        interp += f"전이 증가 범주: {', '.join(transfer_cats)}."
                    st.markdown(interp)
                else:
                    st.info(f"반 {cid}: 상위 범주 정보 없음")

def render_class_network(results):
    """반 전체 네트워크 그래프 (상위 범주 수준 집약) — 반 선택형"""
    metrics = results["metrics_df"]
    extractions = results["extractions"]
    from collections import Counter, defaultdict
    import plotly.graph_objects as go
    import math

    cat_path = OUTPUT_DIR / "type_categories.json"
    if not cat_path.exists():
        st.warning("type_categories.json이 없습니다. pipeline.py를 먼저 실행하세요.")
        return
    import json
    with open(cat_path, "r", encoding="utf-8") as f:
        type_categories = json.load(f)

    all_class_ids = sorted(metrics["class_id"].unique())

    st.markdown("### 반 전체 학습경험 네트워크")
    st.markdown(
        '<div style="background:linear-gradient(135deg,#EBF5FB,#D6EAF8);padding:14px 18px;border-radius:10px;margin-bottom:16px;font-size:.88rem;color:#1B4F72;border-left:4px solid #2E86C1">'
        '<b>🕸️ 안내:</b> 각 반의 상위 범주를 노드로, 범주 간 연결을 엣지로 집약한 네트워크입니다. '
        '노드 크기는 해당 범주의 엔티티 수, 엣지 굵기는 범주 간 관계 수를 나타냅니다. '
        '<span style="color:#E74C3C;font-weight:bold">빨간 엣지</span>는 에필로그→활용 교차 연결(전이 흐름), '
        '<span style="color:#AAA">회색 엣지</span>는 동일 출처 내 연결입니다. '
        '엣지 라벨의 괄호 안 숫자는 전이 연결 수입니다.</div>', unsafe_allow_html=True)

    # 반 선택
    cid = st.selectbox("반 선택", all_class_ids, key="class_network_select")

    # 이 반의 범주 매핑 구축
    cats_data = type_categories.get(cid, [])
    type_to_cat = {}
    for cat_info in cats_data:
        cat_name = cat_info.get("category", "기타")
        for member in cat_info.get("members", []):
            type_to_cat[member] = cat_name

    # 이 반의 엔티티/관계 수집
    cd = metrics[metrics["class_id"] == cid]
    all_ents = []
    all_rels = []
    for sid in cd["student_id"]:
        ext = extractions.get(sid, {"entities": [], "relations": []})
        all_ents.extend(ext.get("entities", []))
        all_rels.extend(ext.get("relations", []))

    for e in all_ents:
        e["_cat"] = type_to_cat.get(e.get("type", ""), "기타")

    ent_by_id = {e["id"]: e for e in all_ents}
    cat_counts = Counter(e["_cat"] for e in all_ents)

    # 범주 간 관계 집계
    cat_edges = defaultdict(lambda: {"total": 0, "cross": 0})
    for r in all_rels:
        src_ent = ent_by_id.get(r.get("source"))
        tgt_ent = ent_by_id.get(r.get("target"))
        if not src_ent or not tgt_ent: continue
        src_cat = src_ent["_cat"]
        tgt_cat = tgt_ent["_cat"]
        if src_cat == tgt_cat: continue

        edge_key = tuple(sorted([src_cat, tgt_cat]))
        cat_edges[edge_key]["total"] += 1

        src_source = src_ent.get("source", "")
        tgt_source = tgt_ent.get("source", "")
        if (src_source.startswith("에필로그") and tgt_source.startswith("활용")) or \
           (src_source.startswith("활용") and tgt_source.startswith("에필로그")):
            cat_edges[edge_key]["cross"] += 1

    cat_names = [c for c in cat_counts if c != "기타"]
    if not cat_names:
        st.info(f"반 {cid}: 범주 정보 없음")
        return

    n_cats = len(cat_names)
    cat_pos = {}
    for ci, cat in enumerate(cat_names):
        angle = 2 * math.pi * ci / n_cats - math.pi / 2
        cat_pos[cat] = (math.cos(angle) * 3, math.sin(angle) * 3)

    fig = go.Figure()

    # 엣지
    for (c1, c2), info in cat_edges.items():
        if c1 not in cat_pos or c2 not in cat_pos: continue
        x0, y0 = cat_pos[c1]
        x1, y1 = cat_pos[c2]
        total = info["total"]
        cross = info["cross"]

        if cross > 0:
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                line=dict(width=min(cross * 2, 12), color='#E74C3C'),
                hovertext=f"{c1} ↔ {c2}<br>전이 연결: {cross}건<br>전체: {total}건", hoverinfo='text',
                showlegend=False))
        if total - cross > 0:
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                line=dict(width=min((total - cross) * 1.0, 10), color='#CCC'),
                hovertext=f"{c1} ↔ {c2}<br>내부 연결: {total - cross}건", hoverinfo='text',
                showlegend=False))

        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        label = f"{total}" if cross == 0 else f"{total}({cross})"
        fig.add_trace(go.Scatter(
            x=[mx], y=[my], mode='text',
            text=[label], textfont=dict(size=11, color='#C0392B' if cross > 0 else '#999'),
            showlegend=False, hoverinfo='skip'))

    # 노드
    cat_palette = ["#4A90D9", "#E8913A", "#2ECC71", "#9B59B6", "#F39C12",
                  "#1ABC9C", "#E74C3C", "#3498DB", "#E67E22", "#27AE60"]
    xs = [cat_pos[c][0] for c in cat_names]
    ys = [cat_pos[c][1] for c in cat_names]
    sizes = [max(35, cat_counts.get(c, 0) * 1.5) for c in cat_names]
    colors_list = [cat_palette[ci % len(cat_palette)] for ci in range(n_cats)]
    texts = [f"{c}<br>({cat_counts.get(c, 0)}개)" for c in cat_names]
    hovers = [f"<b>{c}</b><br>엔티티: {cat_counts.get(c, 0)}개" for c in cat_names]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers+text',
        marker=dict(size=sizes, color=colors_list, line=dict(width=2, color='white')),
        text=texts, textposition="top center",
        textfont=dict(size=11, color='#333'),
        hovertext=hovers, hoverinfo='text',
        showlegend=False))

    fig.update_layout(
        height=650, showlegend=False,
        title=dict(text=f"반 {cid} ({len(cd)}명) — 상위 범주 간 연결 구조", font=dict(size=15)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20))

    st.plotly_chart(fig, use_container_width=True, key=f"class_net_{cid}")

    # ═══════════════════════════════════════════
    # 상세 해석
    # ═══════════════════════════════════════════
    total_cross = sum(v["cross"] for v in cat_edges.values())
    total_inner = sum(v["total"] - v["cross"] for v in cat_edges.values())
    total_all = total_cross + total_inner

    # 범주별 엔티티 수 정렬
    sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
    top_cat = sorted_cats[0] if sorted_cats else ("없음", 0)

    # 전이 경로 정렬
    cross_edges_sorted = sorted(cat_edges.items(), key=lambda x: -x[1]["cross"])
    top_cross_paths = [(k, v) for k, v in cross_edges_sorted if v["cross"] > 0]

    # 연결 없는 범주 (고립 범주)
    connected_cats = set()
    for (c1, c2) in cat_edges:
        connected_cats.add(c1)
        connected_cats.add(c2)
    isolated_cats = [c for c in cat_names if c not in connected_cats]

    # 에필로그/활용별 범주 분포
    epi_cat_counts = Counter(e["_cat"] for e in all_ents if e.get("source", "").startswith("에필로그"))
    app_cat_counts = Counter(e["_cat"] for e in all_ents if e.get("source", "").startswith("활용"))

    # 해석 텍스트 생성
    st.markdown(f"#### 반 {cid} 네트워크 해석")

    interp_lines = []
    interp_lines.append(f"**기본 구조**: {len(cd)}명의 학습경험에서 {len(all_ents)}개 엔티티가 {n_cats}개 상위 범주로 분류되었으며, 범주 간 {total_all}건의 연결이 형성되었습니다.")
    interp_lines.append(f"**핵심 범주**: 가장 많은 엔티티가 속한 범주는 '{top_cat[0]}'({top_cat[1]}개)이며, 이 반 학습자들의 성찰에서 가장 빈번하게 나타나는 학습경험 영역입니다.")

    if total_cross > 0:
        cross_pct = total_cross / total_all * 100 if total_all > 0 else 0
        interp_lines.append(f"**전이 흐름**: 범주 간 전체 연결 {total_all}건 중 {total_cross}건({cross_pct:.0f}%)이 에필로그→활용 전이 연결입니다.")
        if top_cross_paths:
            paths_text = ", ".join(f"'{k[0]}↔{k[1]}'({v['cross']}건)" for k, v in top_cross_paths[:3])
            interp_lines.append(f"**주요 전이 경로**: {paths_text}. 이 경로들은 학습자들이 회고한 경험에서 적용 구상으로 가장 활발하게 연결되는 범주 조합입니다.")
    else:
        interp_lines.append("**전이 흐름**: 범주 간 전이 연결이 관찰되지 않았습니다. 에필로그와 활용의 학습경험이 범주 수준에서 분리되어 있습니다.")

    if isolated_cats:
        interp_lines.append(f"**고립 범주**: {', '.join(isolated_cats)}는 다른 범주와 연결되지 않은 독립적 영역입니다. 이 범주의 학습경험이 다른 영역과 통합되지 않고 있음을 시사합니다.")

    # 에필로그 vs 활용 비중 차이가 큰 범주
    shift_cats = []
    for cat in cat_names:
        epi_n = epi_cat_counts.get(cat, 0)
        app_n = app_cat_counts.get(cat, 0)
        total_n = epi_n + app_n
        if total_n >= 5:
            if app_n > epi_n * 2:
                shift_cats.append(f"'{cat}'(활용 {app_n} > 에필로그 {epi_n})")
            elif epi_n > app_n * 2:
                shift_cats.append(f"'{cat}'(에필로그 {epi_n} > 활용 {app_n})")
    if shift_cats:
        interp_lines.append(f"**출처 편중**: {', '.join(shift_cats)}에서 에필로그/활용 간 비중 차이가 큽니다. 이는 해당 범주가 회고 또는 적용 구상 중 한쪽에 집중되어 있음을 의미합니다.")

    # 수업 설계 제언
    interp_lines.append("")
    interp_lines.append("**📋 수업 설계 시사점**:")
    if total_cross > 0 and top_cross_paths:
        best_path = top_cross_paths[0]
        interp_lines.append(f"- 가장 활발한 전이 경로인 '{best_path[0][0]}↔{best_path[0][1]}'를 강화하는 활동을 설계하면 학습 전이를 촉진할 수 있습니다.")
    if isolated_cats:
        interp_lines.append(f"- 고립 범주({', '.join(isolated_cats)})를 다른 범주와 연결하는 성찰 프롬프트나 활동을 추가하면 통합적 학습이 촉진될 수 있습니다.")
    zero_transfer = len(cd[cd["cross_ratio"] == 0])
    if zero_transfer > 0:
        interp_lines.append(f"- 이 반의 전이 부재 학생 {zero_transfer}명에 대해 '학습 내용을 전공·일상에 어떻게 적용할 수 있는지' 성찰을 유도하는 보완이 필요합니다.")

    st.markdown("\n\n".join(interp_lines))

def render_class_analysis(results):
    metrics = results["metrics_df"]; graph_results = results["graph_results"]
    extractions = results["extractions"]
    from collections import Counter, defaultdict

    all_class_ids = sorted(metrics["class_id"].unique())

    # ══════════════════════════════════════════════════════
    # 반 선택 세부 분석
    # ══════════════════════════════════════════════════════
    class_id = st.selectbox("반 선택 (세부 분석)", all_class_ids, key="class_analysis_select")
    class_data = metrics[metrics["class_id"] == class_id]
    gr = graph_results.get(class_id, {}); labels = gr.get("community_labels", {}); communities = gr.get("communities", {})

    # ── 도출된 학습경험 범주 ──
    if labels:
        st.markdown(f"### 반 {class_id}에서 도출된 학습경험 범주 ({len(labels)}개)")
        for cid_label, label in labels.items():
            n = len(communities.get(cid_label, []))
            st.markdown(f'<span class="category-tag">범주 {cid_label}: {label} ({n}개 노드)</span>', unsafe_allow_html=True)
        st.markdown("")

    # ── 에필로그/활용별 유형 수집 (상위 범주 포함) ──
    epi_by_cat = defaultdict(list)  # {상위범주: [{"type": ..., "text": ..., "학생": ...}, ...]}
    app_by_cat = defaultdict(list)

    for _, row in class_data.iterrows():
        sid = row["student_id"]
        short_name = sid.split("_")[-1]
        ext = extractions.get(sid, {"entities": []})
        for ent in ext.get("entities", []):
            etype = ent.get("type", "")
            etext = ent.get("text", "")
            esource = ent.get("source", "")
            ecat = ent.get("category", "기타")
            detail = {"학생": short_name, "전공": row["major"], "유형": etype, "내용": etext}

            if esource.startswith("에필로그"):
                epi_by_cat[ecat].append(detail)
            elif esource.startswith("활용"):
                app_by_cat[ecat].append(detail)

    # ── 에필로그 학습경험 요소 (연구문제 1-1) ──
    st.markdown(f"### 에필로그(회고) 학습경험 요소")
    st.caption("연구문제 1-1: 에필로그에 나타난 학습경험 요소의 유형과 분포 특성")

    epi_cat_sorted = sorted(epi_by_cat.keys(), key=lambda x: -len(epi_by_cat[x]))
    if epi_cat_sorted:
        for cat_name in epi_cat_sorted:
            items = epi_by_cat[cat_name]
            type_counter = Counter(d["유형"] for d in items)
            n_students = len(set(d["학생"] for d in items))
            type_summary = ", ".join(f"{t}({c})" for t, c in type_counter.most_common(5))
            pct = len(items) / sum(len(v) for v in epi_by_cat.values()) * 100

            # 상위 범주 헤더 (크게 강조)
            st.markdown(
                f'<div style="background:#EBF3FB; border-left:4px solid #4A90D9; padding:10px 14px; margin:8px 0 4px 0; border-radius:4px;">'
                f'<span style="font-size:17px; font-weight:bold; color:#1A3764;">📘 {cat_name}</span>'
                f'<span style="font-size:14px; color:#555; margin-left:12px;">{len(items)}개 요소 · {n_students}명 · {pct:.0f}%</span>'
                f'</div>', unsafe_allow_html=True)

            with st.expander(f"세부 유형: {type_summary}"):
                for type_name, count in type_counter.most_common():
                    type_items = [d for d in items if d["유형"] == type_name]
                    st.markdown(f"**{type_name}** ({count}개)")
                    for d in type_items:
                        st.markdown(f"- {d['학생']}({d['전공']}): {d['내용']}")
                    st.markdown("")
    else:
        st.info("에필로그 요소가 없습니다.")

    # ── 활용 학습경험 요소 (연구문제 1-2) ──
    st.markdown("---")
    st.markdown(f"### 활용(적용 구상) 학습 전이 요소")
    st.caption("연구문제 1-2: 활용에 나타난 학습 전이의 유형과 특징")

    app_cat_sorted = sorted(app_by_cat.keys(), key=lambda x: -len(app_by_cat[x]))
    if app_cat_sorted:
        for cat_name in app_cat_sorted:
            items = app_by_cat[cat_name]
            type_counter = Counter(d["유형"] for d in items)
            n_students = len(set(d["학생"] for d in items))
            type_summary = ", ".join(f"{t}({c})" for t, c in type_counter.most_common(5))
            pct = len(items) / sum(len(v) for v in app_by_cat.values()) * 100

            # 상위 범주 헤더 (크게 강조)
            st.markdown(
                f'<div style="background:#FDF3E7; border-left:4px solid #E8913A; padding:10px 14px; margin:8px 0 4px 0; border-radius:4px;">'
                f'<span style="font-size:17px; font-weight:bold; color:#8B5E2B;">📙 {cat_name}</span>'
                f'<span style="font-size:14px; color:#555; margin-left:12px;">{len(items)}개 요소 · {n_students}명 · {pct:.0f}%</span>'
                f'</div>', unsafe_allow_html=True)

            with st.expander(f"세부 유형: {type_summary}"):
                for type_name, count in type_counter.most_common():
                    type_items = [d for d in items if d["유형"] == type_name]
                    st.markdown(f"**{type_name}** ({count}개)")
                    for d in type_items:
                        st.markdown(f"- {d['학생']}({d['전공']}): {d['내용']}")
                    st.markdown("")
    else:
        st.info("활용 요소가 없습니다.")

    # ── 전이 현황 (연구문제 1-3) ──
    st.markdown("---")
    st.markdown(f"### 전이 현황")
    st.caption("연구문제 1-3: 에필로그와 활용 간의 의미적 관계 구조")

    total = len(class_data)
    with_transfer = len(class_data[class_data["cross_ratio"] > 0])
    without_transfer = total - with_transfer
    st.markdown(f"- 전이 있음: **{with_transfer}명** ({with_transfer/total*100:.0f}%)")
    st.markdown(f"- 전이 없음: **{without_transfer}명** ({without_transfer/total*100:.0f}%)")

    if without_transfer > 0:
        no_transfer_names = class_data[class_data["cross_ratio"] == 0]["student_id"].apply(lambda x: x.split("_")[-1]).tolist()
        st.markdown(f"- ⚠️ 전이 부재 학생: {', '.join(no_transfer_names)}")

def render_student_gauge(row, results):
    """반 내 위치 게이지"""
    structure_pct = row.get("relation_entity_ratio_pct", 0)
    transfer_pct = row.get("cross_ratio_pct", 0)

    import plotly.graph_objects as go
    gauge_cols = st.columns(2)
    with gauge_cols[0]:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number", value=round(structure_pct, 1),
            title={"text": "성찰 구조 백분위", "font": {"size": 16}},
            number={"font": {"size": 36}},
            gauge={"axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1A3764", "thickness": 0.3}, "bgcolor": "white",
                "steps": [{"range": [0, 25], "color": "#FFE0E0"}, {"range": [25, 50], "color": "#FFF3E0"},
                          {"range": [50, 75], "color": "#E8F5E9"}, {"range": [75, 100], "color": "#E3F2FD"}],
                "threshold": {"line": {"color": "#E74C3C", "width": 2}, "thickness": 0.8, "value": 50}}))
        fig_g1.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_g1, use_container_width=True)
    with gauge_cols[1]:
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number", value=round(transfer_pct, 1),
            title={"text": "전이 연결 백분위", "font": {"size": 16}},
            number={"font": {"size": 36}},
            gauge={"axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#E8913A", "thickness": 0.3}, "bgcolor": "white",
                "steps": [{"range": [0, 25], "color": "#FFE0E0"}, {"range": [25, 50], "color": "#FFF3E0"},
                          {"range": [50, 75], "color": "#E8F5E9"}, {"range": [75, 100], "color": "#E3F2FD"}],
                "threshold": {"line": {"color": "#E74C3C", "width": 2}, "thickness": 0.8, "value": 50}}))
        fig_g2.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_g2, use_container_width=True)

    if structure_pct >= 50 and transfer_pct >= 50:
        st.success("반 내 상위권. 성찰 구조와 전이 연결 모두 양호합니다.")
    elif structure_pct >= 50:
        st.warning("성찰 구조는 양호하나, 학습 전이 연결이 부족합니다.")
    elif transfer_pct >= 50:
        st.info("전이 연결은 관찰되나, 성찰 구조가 다소 느슨합니다.")
    else:
        st.error("성찰 구조와 전이 연결 모두 보완이 필요합니다.")

    # 진단 코멘트
    metrics = results["metrics_df"]
    class_data = metrics[metrics["class_id"] == row["class_id"]]
    rer_med = class_data["relation_entity_ratio"].median()
    cr_med = class_data["cross_ratio"].median()
    if row["relation_entity_ratio"] >= rer_med and row["cross_ratio"] >= cr_med:
        st.markdown("**포지션**: 구조적 밀도 높음 + 전이 활발. 학습경험을 밀접하게 연결하면서 전공·일상 적용도 구체적으로 구상하고 있습니다.")
    elif row["relation_entity_ratio"] < rer_med and row["cross_ratio"] >= cr_med:
        st.markdown("**포지션**: 구조적 밀도 낮음 + 전이 활발. 경험 요소 간 연결을 더 풍부하게 형성하면 성찰이 심화될 수 있습니다.")
    elif row["relation_entity_ratio"] >= rer_med:
        st.markdown("**포지션**: 구조적 밀도 높음 + 전이 부족. 배운 것을 전공이나 일상에 어떻게 적용할 수 있는지 구체적으로 연결지어 보면 좋겠습니다.")
    else:
        st.markdown("**포지션**: 구조적 밀도 낮음 + 전이 부족. 경험 간 연결을 형성하고, 전공·일상 적용을 구체화해 보세요.")

def render_student_graph(student_id, ext, results):
    """학습경험 네트워크 그래프"""
    entities = ext.get("entities", []); relations = ext.get("relations", [])

    if entities:
        import plotly.graph_objects as go
        import random, math
        from collections import Counter as _GCtr, defaultdict as _Gdd

        epi_ids = {e["id"] for e in entities if e.get("source", "").startswith("에필로그")}
        app_ids = {e["id"] for e in entities if e.get("source", "").startswith("활용")}

        # ── 종합의견 생성 ──
        n_ent = len(entities)
        n_rel = len(relations)
        cross_rels_list = [r for r in relations
                          if (r["source"] in epi_ids and r["target"] in app_ids)
                          or (r["source"] in app_ids and r["target"] in epi_ids)]
        n_cross = len(cross_rels_list)

        # 허브 노드 찾기
        conn_count = _Gdd(int)
        for r in relations:
            conn_count[r["source"]] += 1
            conn_count[r["target"]] += 1
        hub_id = max(conn_count, key=conn_count.get) if conn_count else None
        hub_node = next((e for e in entities if e["id"] == hub_id), None)
        hub_text = f"'{hub_node.get('text','')}'({hub_node.get('type','')})" if hub_node else "없음"
        hub_conn = conn_count.get(hub_id, 0)

        # 밀도 판단
        density = n_rel / (n_ent * (n_ent - 1)) if n_ent > 1 else 0
        if density >= 0.25:
            density_text = "그래프 밀도가 높아, 주요 경험 요소 간 연결이 형성되어 있습니다."
        elif density >= 0.15:
            density_text = "그래프 밀도가 적정 수준으로, 주요 경험 요소 간 연결이 형성되어 있습니다."
        else:
            density_text = "그래프 밀도가 낮아, 경험 요소 간 연결이 다소 분절적입니다."

        # 정서 전환
        epi_cat_list = [e.get("category", "") for e in entities if e.get("source", "").startswith("에필로그")]
        app_cat_list = [e.get("category", "") for e in entities if e.get("source", "").startswith("활용")]

        summary = f"총 {n_ent}개 엔티티와 {n_rel}개 관계가 추출되었으며, {hub_text}이(가) {hub_conn}개의 연결을 가진 핵심 허브 노드입니다. {density_text}"
        if n_cross > 0:
            summary += f" 에필로그→활용 전이 경로가 {n_cross}개 형성되어, 학습 전이가 관찰됩니다."
        else:
            summary += " 에필로그→활용 전이 경로가 형성되지 않았습니다."

        # 종합의견 박스
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#EBF5FB,#D6EAF8);padding:14px 18px;border-radius:10px;margin-bottom:12px;font-size:.88rem;color:#1B4F72;border-left:4px solid #2E86C1">'
            f'<b>🕸️ 종합의견:</b> {summary}</div>', unsafe_allow_html=True)

        # ── 범례 태그 ──
        cat_colors = {
            "학습활동및과정": "#4A90D9", "학습성과및역량": "#2ECC71", "정의적변화": "#E74C3C",
            "협력및사회적경험": "#9B59B6", "전공및일상전이": "#F39C12", "학습동기및목표": "#1ABC9C",
        }
        # 이 학생의 실제 범주 수집
        all_cats = set()
        for e in entities:
            all_cats.add(e.get("category", "기타"))

        used_colors = {}
        default_palette = ["#4A90D9", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C", "#E67E22", "#3498DB"]
        for i, cat in enumerate(sorted(all_cats)):
            used_colors[cat] = cat_colors.get(cat, default_palette[i % len(default_palette)])

        legend_html = " ".join(
            f'<span style="display:inline-block;padding:5px 16px;border-radius:20px;font-weight:700;color:white;font-size:.9rem;background:{used_colors[c]};margin:2px">{c}</span>'
            for c in sorted(all_cats) if c != "기타"
        )
        st.markdown(f"<div style='margin-bottom:10px'>{legend_html}</div>", unsafe_allow_html=True)

        # ── Plotly 그래프 생성 ──
        random.seed(42)

        # 레이아웃: 에필로그를 왼쪽, 활용을 오른쪽에 배치 (간격 축소)
        pos = {}
        epi_ents = [e for e in entities if e.get("source", "").startswith("에필로그")]
        app_ents = [e for e in entities if e.get("source", "").startswith("활용")]

        for i, e in enumerate(epi_ents):
            angle = (i / max(len(epi_ents), 1)) * math.pi + math.pi / 2
            r = 1.0 + random.uniform(-0.2, 0.2)
            pos[e["id"]] = (-1.0 + r * math.cos(angle), r * math.sin(angle))

        for i, e in enumerate(app_ents):
            angle = (i / max(len(app_ents), 1)) * math.pi - math.pi / 2
            r = 1.0 + random.uniform(-0.2, 0.2)
            pos[e["id"]] = (1.0 + r * math.cos(angle), r * math.sin(angle))

        traces = []

        # 엣지
        for r in relations:
            if r["source"] in pos and r["target"] in pos:
                x0, y0 = pos[r["source"]]
                x1, y1 = pos[r["target"]]
                is_cross = (r["source"] in epi_ids and r["target"] in app_ids) or \
                           (r["source"] in app_ids and r["target"] in epi_ids)
                edge_color = "#E74C3C" if is_cross else "#CCC"
                edge_width = 3 if is_cross else 1.5

                traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='skip', showlegend=False))

                # 엣지 라벨
                rel_type = r.get("type", "")
                if rel_type:
                    label_color = "#C0392B" if is_cross else "#999"
                    traces.append(go.Scatter(
                        x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode='text',
                        text=[rel_type], textfont=dict(size=10, color=label_color),
                        showlegend=False, hoverinfo='skip'))

        # 노드 (범주별)
        for cat in sorted(all_cats):
            cat_ents = [e for e in entities if e.get("category", "기타") == cat]
            if not cat_ents:
                continue
            color = used_colors.get(cat, "#888")
            xs = [pos[e["id"]][0] for e in cat_ents if e["id"] in pos]
            ys = [pos[e["id"]][1] for e in cat_ents if e["id"] in pos]
            valid_ents = [e for e in cat_ents if e["id"] in pos]

            sizes = []
            for e in valid_ents:
                conn = sum(1 for r in relations if r["source"] == e["id"] or r["target"] == e["id"])
                sizes.append(25 + conn * 8)

            # 노드 라벨: 세부유형 (위) + 구체적 내용 (아래)
            texts = []
            for e in valid_ents:
                etype = e.get("type", "")
                etext = e.get("text", "")
                if len(etext) > 18:
                    etext = etext[:18] + "..."
                texts.append(f"{etype}<br><sub>{etext}</sub>")

            hover_texts = []
            for e in valid_ents:
                source_label = "에필로그" if e.get("source", "").startswith("에필로그") else "활용"
                hover_texts.append(
                    f"<b>{e.get('type', '')}</b><br>"
                    f"범주: {cat}<br>"
                    f"출처: {source_label}<br>"
                    f"내용: {e.get('text', '')}")

            traces.append(go.Scatter(
                x=xs, y=ys, mode='markers+text',
                marker=dict(size=sizes, color=color, line=dict(width=2, color='white')),
                text=texts, textposition="top center",
                textfont=dict(size=11, color='#333'),
                hovertext=hover_texts, hoverinfo='text',
                name=cat))

        fig = go.Figure(data=traces)
        fig.update_layout(
            height=610, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig, use_container_width=True, key=f"graph_{student_id}")

    else:
        st.info("추출된 엔티티가 없습니다.")

def render_student_elements(student_id, row, ext, results):
    """추출된 학습경험 요소 + 원문"""
    # 메트릭 카드
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: st.metric("엔티티 수", row["n_entities"])
    with mc2: st.metric("관계 수", row["n_relations"])
    with mc3: st.metric("에필로그 요소", row["n_epi_entities"])
    with mc4: st.metric("활용 요소", row["n_app_entities"])

    entities = ext.get("entities", []); relations = ext.get("relations", [])
    epi_ents = [e for e in entities if e.get("source", "").startswith("에필로그")]
    app_ents = [e for e in entities if e.get("source", "").startswith("활용")]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**에필로그에서 추출된 요소**")
        for e in epi_ents: st.markdown(f"- **{e.get('type', '')}**: {e.get('text', '')}")
    with c2:
        st.markdown("**활용에서 추출된 요소**")
        for e in app_ents: st.markdown(f"- **{e.get('type', '')}**: {e.get('text', '')}")

    # 원문 표시
    df = results["df"]
    student_row = df[df["student_id"] == student_id]
    if not student_row.empty:
        sr = student_row.iloc[0]
        with st.expander("📄 학생 작성 원문 보기", expanded=True):
            st.markdown("**에필로그 (회고)**")
            st.markdown(f'<div style="background:#F8F9FA;padding:12px;border-radius:8px;font-size:.9rem;line-height:1.7;border-left:3px solid #4A90D9">{sr.get("epilogue","")}</div>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown("**활용 (적용 구상)**")
            st.markdown(f'<div style="background:#FFF8F0;padding:12px;border-radius:8px;font-size:.9rem;line-height:1.7;border-left:3px solid #E8913A">{sr.get("application","")}</div>', unsafe_allow_html=True)

def render_student_transfer(student_id, row, ext, results):
    """전이 분석"""
    entities = ext.get("entities", []); relations = ext.get("relations", [])
    epi_ents = [e for e in entities if e.get("source", "").startswith("에필로그")]
    app_ents = [e for e in entities if e.get("source", "").startswith("활용")]
    epi_ids = {e["id"] for e in epi_ents}; app_ids = {e["id"] for e in app_ents}
    id_to_text = {e["id"]: e.get("text", "") for e in entities}
    cross_rels = [r for r in relations if (r["source"] in epi_ids and r["target"] in app_ids) or (r["source"] in app_ids and r["target"] in epi_ids)]
    if cross_rels:
        st.markdown("### 에필로그 → 활용 전이 경로")
        for rel in cross_rels:
            st.markdown(f'<div class="transfer-path">"{id_to_text.get(rel["source"], "")}" →<b>({rel.get("type", "")})</b>→ "{id_to_text.get(rel["target"], "")}"</div>', unsafe_allow_html=True)
    else:
        st.info("에필로그→활용 교차 연결이 없습니다.")

    # 진단 코멘트
    metrics = results["metrics_df"]
    class_data = metrics[metrics["class_id"] == row["class_id"]]
    rer_med = class_data["relation_entity_ratio"].median()
    cr_med = class_data["cross_ratio"].median()
    if row["relation_entity_ratio"] >= rer_med and row["cross_ratio"] >= cr_med:
        st.success("**포지션: 구조적 밀도 높음 + 전이 활발**\n\n학습경험을 밀접하게 연결하면서 전공·일상 적용도 구체적으로 구상하고 있습니다.")
    elif row["relation_entity_ratio"] < rer_med and row["cross_ratio"] >= cr_med:
        st.info("**포지션: 구조적 밀도 낮음 + 전이 활발**\n\n경험 요소 간 연결을 더 풍부하게 형성하면 성찰이 심화될 수 있습니다.")
    elif row["relation_entity_ratio"] >= rer_med:
        st.warning("**포지션: 구조적 밀도 높음 + 전이 부족**\n\n배운 것을 전공이나 일상에 어떻게 적용할 수 있는지 구체적으로 연결지어 보면 좋겠습니다.")
    else:
        st.error("**포지션: 구조적 밀도 낮음 + 전이 부족**\n\n경험 간 연결을 더 형성하고, 전공·일상 적용을 구체화해 보세요.")

def render_major_positioning_map(results):
    """전공별 포지셔닝 맵"""
    metrics = results["metrics_df"]
    import plotly.graph_objects as go

    st.markdown("### 전공별 포지셔닝 맵")

    all_majors = sorted(metrics["major"].unique())
    major_colors = {}
    palette = ["#4A90D9", "#E8913A", "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12", "#1ABC9C", "#E67E22",
               "#3498DB", "#C0392B", "#27AE60", "#8E44AD", "#D35400", "#2980B9", "#16A085"]
    for i, m in enumerate(all_majors):
        major_colors[m] = palette[i % len(palette)]

    # 종합의견
    major_stats = []
    for major in all_majors:
        md = metrics[metrics["major"] == major]
        if len(md) < 2: continue
        major_stats.append({"major": major, "n": len(md), "cr": md["cross_ratio"].mean(), "rer": md["relation_entity_ratio"].mean()})

    if major_stats:
        sorted_cr = sorted(major_stats, key=lambda x: -x["cr"])
        high = sorted_cr[0] if sorted_cr else None
        low = sorted_cr[-1] if sorted_cr else None
        summary = f"전공별 포지셔닝 맵입니다. "
        if high and low:
            summary += f"전이 연결이 가장 활발한 전공은 {high['major']}({high['cr']:.3f}), 가장 낮은 전공은 {low['major']}({low['cr']:.3f})입니다."
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#FEF9E7,#FCF3CF);padding:14px 18px;border-radius:10px;margin-bottom:12px;font-size:.88rem;color:#7D6608;border-left:4px solid #F39C12">'
            f'<b>📝 종합의견:</b> {summary}</div>', unsafe_allow_html=True)

    # 축 범위 통일
    x_min = max(0, metrics["relation_entity_ratio"].min() - 0.05)
    x_max = min(1.2, metrics["relation_entity_ratio"].max() + 0.05)
    y_min = max(-0.02, metrics["cross_ratio"].min() - 0.02)
    y_max = min(0.5, metrics["cross_ratio"].max() + 0.03)
    x_mid = metrics["relation_entity_ratio"].median()
    y_mid = metrics["cross_ratio"].median()

    n_majors = len(all_majors)
    for i in range(0, n_majors, 2):
        pm_cols = st.columns(2)
        for j, pm_col in enumerate(pm_cols):
            idx = i + j
            if idx >= n_majors: break
            major = all_majors[idx]
            mdata = metrics[metrics["major"] == major]
            color = major_colors.get(major, "#888")

            with pm_col:
                fig = go.Figure()
                fig.add_shape(type="rect", x0=x_min, y0=y_mid, x1=x_mid, y1=y_max, fillcolor="rgba(46,204,113,0.06)", line=dict(width=0))
                fig.add_shape(type="rect", x0=x_mid, y0=y_mid, x1=x_max, y1=y_max, fillcolor="rgba(46,204,113,0.14)", line=dict(width=0))
                fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=x_mid, y1=y_mid, fillcolor="rgba(231,76,60,0.06)", line=dict(width=0))
                fig.add_shape(type="rect", x0=x_mid, y0=y_min, x1=x_max, y1=y_mid, fillcolor="rgba(241,196,15,0.08)", line=dict(width=0))
                fig.add_hline(y=y_mid, line_dash="dot", line_color="#CCC", line_width=1)
                fig.add_vline(x=x_mid, line_dash="dot", line_color="#CCC", line_width=1)

                names = [r["student_id"].split("_")[-1] for _, r in mdata.iterrows()]
                sizes = [max(20, r["n_entities"] * 3.5) for _, r in mdata.iterrows()]
                hovers = [f"<b>{r['student_id'].split('_')[-1]}</b> ({r['class_id']}반)<br>밀도: {r['relation_entity_ratio']:.3f}<br>전이: {r['cross_ratio']:.3f}"
                         for _, r in mdata.iterrows()]

                fig.add_trace(go.Scatter(
                    x=mdata["relation_entity_ratio"], y=mdata["cross_ratio"],
                    mode='markers+text',
                    marker=dict(size=sizes, color=color, line=dict(width=2, color='white'), opacity=0.85),
                    text=names, textposition="top center", textfont=dict(size=9, color='#333'),
                    hovertext=hovers, hoverinfo='text', name=major))

                fig.update_layout(
                    height=380, showlegend=False,
                    title=dict(text=f"{major} ({len(mdata)}명)", font=dict(size=13)),
                    xaxis=dict(title="← 구조적 밀도 →", range=[x_min, x_max], tickformat=".2f", gridcolor="#F0F0F0"),
                    yaxis=dict(title="← 전이 연결 →", range=[y_min, y_max], tickformat=".3f", gridcolor="#F0F0F0"),
                    plot_bgcolor='white', margin=dict(l=60, r=20, t=40, b=60))
                st.plotly_chart(fig, use_container_width=True, key=f"pm_major_{major}")

def render_major_analysis(results):
    metrics = results["metrics_df"]
    extractions = results["extractions"]
    from collections import Counter
    import plotly.graph_objects as go

    st.markdown("### 전공별 학습경험 상위 범주 비교")

    # 종합의견 생성
    all_majors = sorted(metrics["major"].unique())
    major_stats = []
    for major in all_majors:
        md = metrics[metrics["major"] == major]
        cr_mean = md["cross_ratio"].mean()
        rer_mean = md["relation_entity_ratio"].mean()
        cr_zero = len(md[md["cross_ratio"] == 0])

        # 에필로그/활용 범주 분석
        epi_cats = Counter()
        app_cats = Counter()
        for sid in md["student_id"]:
            ext = extractions.get(sid, {"entities": []})
            for ent in ext.get("entities", []):
                cat = ent.get("category", "기타")
                if ent.get("source", "").startswith("에필로그"):
                    epi_cats[cat] += 1
                elif ent.get("source", "").startswith("활용"):
                    app_cats[cat] += 1

        top_epi = epi_cats.most_common(1)[0][0] if epi_cats else "없음"
        top_app = app_cats.most_common(1)[0][0] if app_cats else "없음"
        major_stats.append({"major": major, "n": len(md), "cr": cr_mean, "rer": rer_mean, "zero": cr_zero, "top_epi": top_epi, "top_app": top_app})

    # 전이가 높은/낮은 전공
    if major_stats:
        sorted_by_cr = sorted(major_stats, key=lambda x: -x["cr"])
        high_majors = [s for s in sorted_by_cr if s["cr"] > 0.15 and s["n"] >= 2]
        low_majors = [s for s in sorted_by_cr if s["cr"] < 0.10 and s["n"] >= 2]

        summary_lines = []
        summary_lines.append(f"총 {len(all_majors)}개 전공의 학생이 참여하였습니다.")

        if high_majors:
            high_names = ", ".join(f"{s['major']}({s['n']}명, 전이 {s['cr']:.3f})" for s in high_majors[:3])
            summary_lines.append(f"전이 연결이 활발한 전공: {high_names}.")

        if low_majors:
            low_names = ", ".join(f"{s['major']}({s['n']}명, 전이 {s['cr']:.3f})" for s in low_majors[:3])
            summary_lines.append(f"전이 연결이 낮은 전공: {low_names}.")

        # 전공별 성찰 초점 차이
        epi_focus = set(s["top_epi"] for s in major_stats if s["n"] >= 2)
        if len(epi_focus) > 1:
            summary_lines.append(f"전공에 따라 성찰의 초점이 다르게 나타납니다. 이는 전공 배경이 학습경험의 의미 부여 방식에 영향을 줄 수 있음을 시사합니다.")

        total_zero = sum(s["zero"] for s in major_stats)
        if total_zero > 0:
            summary_lines.append(f"전이 부재 학생 {total_zero}명은 전공과 무관하게 분포하며, 전공 특성보다 개인 차이에 기인하는 것으로 보입니다.")

        st.markdown(
            f'<div style="background:linear-gradient(135deg,#FEF9E7,#FCF3CF);padding:14px 18px;border-radius:10px;margin-bottom:16px;font-size:.88rem;color:#7D6608;border-left:4px solid #F39C12">'
            f'<b>📝 종합의견:</b> {_md_bold_to_html("  ".join(summary_lines))}</div>', unsafe_allow_html=True)

    st.caption("각 전공 내 에필로그/활용 요소의 비율(%)로 표시합니다. 인원 3명 미만 전공은 *로 표시합니다.")

    # 전체 전공의 비율 데이터 수집 (X축 통일용)
    all_major_data = {}
    for major in all_majors:
        md = metrics[metrics["major"] == major]
        epi_cats = Counter()
        app_cats = Counter()
        for sid in md["student_id"]:
            ext = extractions.get(sid, {"entities": []})
            for ent in ext.get("entities", []):
                cat = ent.get("category", "기타")
                if ent.get("source", "").startswith("에필로그"):
                    epi_cats[cat] += 1
                elif ent.get("source", "").startswith("활용"):
                    app_cats[cat] += 1
        total_epi = sum(epi_cats.values())
        total_app = sum(app_cats.values())
        epi_pct = {c: (v / total_epi * 100 if total_epi > 0 else 0) for c, v in epi_cats.items()}
        app_pct = {c: (v / total_app * 100 if total_app > 0 else 0) for c, v in app_cats.items()}
        all_cats = sorted(set(list(epi_cats.keys()) + list(app_cats.keys())),
                        key=lambda x: -(epi_pct.get(x, 0) + app_pct.get(x, 0)))
        all_major_data[major] = {"epi_pct": epi_pct, "app_pct": app_pct, "all_cats": all_cats, "n": len(md)}

    n_majors = len(all_majors)
    cols_per_row = 2
    for row_start in range(0, n_majors, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= n_majors:
                break
            major = all_majors[idx]
            mdata = all_major_data[major]
            all_cats = mdata["all_cats"]
            n_students = mdata["n"]
            small_flag = "*" if n_students < 3 else ""

            with col:
                if all_cats and all_cats != ["기타"]:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=all_cats, x=[mdata["epi_pct"].get(c, 0) for c in all_cats],
                        orientation='h', name='에필로그', marker_color='#4A90D9', opacity=0.85))
                    fig.add_trace(go.Bar(
                        y=all_cats, x=[mdata["app_pct"].get(c, 0) for c in all_cats],
                        orientation='h', name='활용', marker_color='#E8913A', opacity=0.85))
                    fig.update_layout(
                        barmode='group', height=max(250, len(all_cats) * 35),
                        title=dict(text=f"{major} ({n_students}명){small_flag}", font=dict(size=13)),
                        xaxis=dict(title="비율 (%)"),
                        yaxis=dict(autorange="reversed"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9)),
                        margin=dict(l=10, r=10, t=40, b=30), plot_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True, key=f"major_cat_{major}")

                    top_epi = max(mdata["epi_pct"].items(), key=lambda x: x[1]) if mdata["epi_pct"] else ("없음", 0)
                    top_app = max(mdata["app_pct"].items(), key=lambda x: x[1]) if mdata["app_pct"] else ("없음", 0)

                    interp = f"**{major}**: 에필로그 '{top_epi[0]}'({top_epi[1]:.0f}%) 중심. "
                    interp += f"활용 '{top_app[0]}'({top_app[1]:.0f}%) 중심."
                    if n_students < 3:
                        interp += f" *(인원 {n_students}명으로 해석 주의)*"
                    st.markdown(interp)
                else:
                    st.info(f"{major}: 상위 범주 정보 없음")

def render_validation(results):
    metrics = results["metrics_df"]; from scipy import stats as sp_stats
    st.markdown("### 길이 독립성 검증")
    st.caption("텍스트 길이와 구조적 지표의 Spearman 상관 (|ρ| < 0.3이면 길이 독립적)")
    all_labels = {**METRIC_LABELS, "coverage": "범주 커버리지 (제외됨)"}
    all_cols = FINAL_METRICS + (["coverage"] if "coverage" in metrics.columns else [])
    vdata = []
    for col in all_cols:
        if col in metrics.columns and metrics[col].std() > 0:
            rho, p = sp_stats.spearmanr(metrics["total_len"], metrics[col])
            status = "❌ 제외 (길이 의존적)" if col == "coverage" else ("✅ 독립적" if abs(rho) < LENGTH_CORRELATION_THRESHOLD else "⚠️ 경계")
            vdata.append({"지표": all_labels.get(col, col), "Spearman ρ": f"{rho:.3f}", "p-value": f"{p:.3f}", "판정": status})
    st.dataframe(pd.DataFrame(vdata), use_container_width=True, hide_index=True)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig_v = make_subplots(rows=1, cols=4,
        subplot_titles=[f"{METRIC_LABELS[c]}" for c in FINAL_METRICS])
    for i, col in enumerate(FINAL_METRICS):
        rho_val = 0
        if metrics[col].std() > 0:
            rho_val, _ = sp_stats.spearmanr(metrics["total_len"], metrics[col])
        fig_v.add_trace(go.Scatter(
            x=metrics["total_len"], y=metrics[col], mode='markers',
            marker=dict(size=6, color="#4A90D9", opacity=0.5),
            name=METRIC_LABELS[col], showlegend=False,
            hovertext=[f"{sid}<br>길이: {tl}<br>{METRIC_LABELS[col]}: {v:.3f}"
                      for sid, tl, v in zip(metrics["student_id"], metrics["total_len"], metrics[col])],
            hoverinfo='text'), row=1, col=i+1)
        fig_v.update_xaxes(title_text="텍스트 길이" if i == 0 else "", row=1, col=i+1)
        fig_v.update_yaxes(title_text=f"ρ={rho_val:.3f}", row=1, col=i+1)
    fig_v.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=40), plot_bgcolor='white')
    st.plotly_chart(fig_v, use_container_width=True, key="validation_scatter")
    st.markdown("### 추출 안정성 테스트")
    sp = OUTPUT_DIR / "stability_test_result.json"
    if sp.exists():
        with open(sp, "r", encoding="utf-8") as f: st.json(json.load(f))
    else: st.info("아직 실행하지 않음. 터미널에서 `python stability_test.py` 실행 후 새로고침하세요.")

def render_download(results):
    st.markdown("### 결과 파일 다운로드")
    # 텍스트 파일들
    for label, fname, mime in [("📄 평가 보고서 (.md)", "evaluation_report.md", "text/markdown"), ("🔍 LLM 추출 원본 (.json)", "step1_extractions.json", "application/json"), ("🔗 엔티티 병합 결과 (.json)", "step2_merge_cache.json", "application/json")]:
        p = OUTPUT_DIR / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f: data = f.read()
            st.download_button(label, data, file_name=fname, mime=mime)
    # CSV 파일 (별도 처리)
    for label, fname in [("📊 최종 점수 (.csv)", "final_scores.csv"), ("📊 구조적 지표 (.csv)", "step4_metrics.csv")]:
        p = OUTPUT_DIR / fname
        if p.exists():
            csv_data = pd.read_csv(p).to_csv(index=False).encode("utf-8-sig")
            st.download_button(label, csv_data, file_name=fname, mime="text/csv")

df = None
if uploaded_files: df = load_data_from_uploads(uploaded_files)
else: df = load_data_from_disk()

# data/ 파일이 없어도 outputs/에서 직접 로드 시도
if (df is None or len(df) == 0) and use_existing:
    step1_path = OUTPUT_DIR / "step1_extractions.json"
    metrics_path = OUTPUT_DIR / "final_scores.csv"
    if step1_path.exists() and metrics_path.exists():
        import json as _json
        with open(step1_path, "r", encoding="utf-8") as f:
            _extractions = _json.load(f)
        _metrics_df = pd.read_csv(metrics_path)
        # df를 metrics에서 복원
        df_rows = []
        for _, row in _metrics_df.iterrows():
            sid = row["student_id"]
            ext = _extractions.get(sid, {})
            ents = ext.get("entities", [])
            rels = ext.get("relations", [])
            # 원문은 없지만 메타 정보로 df 구성
            df_rows.append({
                "student_id": sid, "class_id": row["class_id"], "major": row["major"],
                "epilogue": " ".join(e.get("text","") for e in ents if e.get("source","").startswith("에필로그")),
                "application": " ".join(e.get("text","") for e in ents if e.get("source","").startswith("활용")),
                "epi_len": row.get("epi_len", 0), "app_len": row.get("app_len", 0),
                "total_len": row.get("total_len", 0),
            })
        df = pd.DataFrame(df_rows)
        st.sidebar.success(f"✅ 기존 결과에서 {len(df)}명 로드")

if df is None or len(df) == 0:
    st.markdown("""<div style="background:linear-gradient(135deg,#1F4E79,#2E75B6);padding:30px 28px;border-radius:14px;margin-bottom:16px">
        <h2 style="color:white;margin:0">📊 PBL 학습경험 동적 평가 프레임워크</h2>
        <p style="color:#B0D4F1;margin:4px 0 0">GraphRAG 기반 귀납적 평가</p>
    </div>""", unsafe_allow_html=True)
    st.info("👈 사이드바에서 데이터를 업로드하거나, `data/` 폴더에 파일을 배치한 후 새로고침하세요.")
    st.stop()

if "results" not in st.session_state:
    st.markdown(f"### 데이터 확인: {len(df)}명, {df['class_id'].nunique()}개 반")
    with st.expander("데이터 미리보기", expanded=False):
        st.dataframe(df[["student_id", "class_id", "major", "epi_len", "app_len"]].head(10), use_container_width=True, hide_index=True)
    if st.button("🚀 분석 실행", type="primary", use_container_width=True):
        with st.spinner("파이프라인 실행 중..."):
            results = run_pipeline(df, use_existing_extractions=use_existing)
            st.session_state["results"] = results
        st.success("분석 완료!")
        st.rerun()
    st.stop()

results = st.session_state["results"]

# ═══════════════════════════════════════════════════════
# 메뉴별 렌더링
# ═══════════════════════════════════════════════════════

if page == "👤 개인별 보고서":
    metrics = results["metrics_df"]; extractions = results["extractions"]
    graph_results = results["graph_results"]

    col1, col2 = st.columns([1, 3])
    with col1: class_id = st.selectbox("반", sorted(metrics["class_id"].unique()), key="student_class")
    with col2: student_id = st.selectbox("학생", metrics[metrics["class_id"] == class_id]["student_id"].tolist(), key="student_id")
    if not student_id: st.stop()
    row = metrics[metrics["student_id"] == student_id].iloc[0]
    ext = extractions.get(student_id, {"entities": [], "relations": []})

    # 헤더
    st.markdown(f"""<div style="background:linear-gradient(135deg,#1F4E79,#2E75B6);padding:20px 28px;border-radius:14px;margin-bottom:16px">
        <h2 style="color:white;margin:0">📊 개인별 학습경험 분석 보고서</h2>
        <p style="color:#B0D4F1;margin:4px 0 0">{student_id} · {row['major']} · 반 {row['class_id']}</p>
    </div>""", unsafe_allow_html=True)

    # 요약 카드
    structure_pct = row.get("relation_entity_ratio_pct", 0)
    transfer_pct = row.get("cross_ratio_pct", 0)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="mc"><div class="v">{len(ext.get("entities",[]))} / {len(ext.get("relations",[]))}</div><div class="l">엔티티 / 관계</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="mc" style="border-color:#2ECC71"><div class="v">{structure_pct:.0f}%</div><div class="l">성찰 구조 백분위</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="mc" style="border-color:#E8913A"><div class="v">{transfer_pct:.0f}%</div><div class="l">전이 연결 백분위</div></div>', unsafe_allow_html=True)

    # 탭
    tab1, tab2, tab3, tab4 = st.tabs(["📊 반 내 위치", "🕸️ 지식 그래프", "📝 학습경험 요소", "🔄 전이 분석"])

    with tab1:
        render_student_gauge(row, results)

    with tab2:
        render_student_graph(student_id, ext, results)

    with tab3:
        render_student_elements(student_id, row, ext, results)

    with tab4:
        render_student_transfer(student_id, row, ext, results)

elif page == "🏫 반별 보고서":
    st.markdown(f"""<div style="background:linear-gradient(135deg,#1F4E79,#2E75B6);padding:20px 28px;border-radius:14px;margin-bottom:16px">
        <h2 style="color:white;margin:0">🏫 반별 학습경험 분석 보고서</h2>
        <p style="color:#B0D4F1;margin:4px 0 0">{len(results['metrics_df'])}명 · {results['metrics_df']['class_id'].nunique()}개 반 · 반별 성찰 구조 및 전이 특성 비교</p>
    </div>""", unsafe_allow_html=True)

    render_overview_metrics(results)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ 반별 포지셔닝 맵", "🕸️ 반 전체 네트워크", "📊 반별 상위 범주 비교", "📋 반별 세부", "✅ 타당화"])

    with tab1:
        render_positioning_map(results)
    with tab2:
        render_class_network(results)
    with tab3:
        render_category_compare(results)
    with tab4:
        render_class_analysis(results)
    with tab5:
        render_validation(results)

elif page == "🎓 전공별 보고서":
    st.markdown(f"""<div style="background:linear-gradient(135deg,#1F4E79,#2E75B6);padding:20px 28px;border-radius:14px;margin-bottom:16px">
        <h2 style="color:white;margin:0">🎓 전공별 학습경험 분석 보고서</h2>
        <p style="color:#B0D4F1;margin:4px 0 0">{results['metrics_df']['major'].nunique()}개 전공 · 전공별 성찰 범주 및 전이 특성 비교</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🗺️ 전공별 포지셔닝 맵", "📊 전공별 상위 범주 비교", "📥 다운로드"])

    with tab1:
        render_major_positioning_map(results)
    with tab2:
        render_major_analysis(results)
    with tab3:
        render_download(results)
