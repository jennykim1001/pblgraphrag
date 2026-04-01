"""
pipeline.py — 5단계 귀납적 파이프라인

1단계: LLM 자유 추출 (에필로그+활용 동시, source 태깅)
2단계: 엔티티 해상도
  2a: type 정규화 — EDC(Zhang & Soh, 2024) self-canonicalization
  2b: 엔티티 병합 — Graphusion(Yang et al., 2024) knowledge fusion
3단계: 그래프 구성 + 커뮤니티 탐지 (Louvain, random_state=42)
4단계: 구조적 지표 산출 + 지표 선정 (VIF, 길이 독립성)
5단계: 반 내 상대 위치 판정 + 보고서 산출

실행: python pipeline.py
"""
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import networkx as nx
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats

from config import *


# ═══════════════════════════════════════════════════════════
# 0. 데이터 로드
# ═══════════════════════════════════════════════════════════
def load_all_data() -> pd.DataFrame:
    """모든 반 데이터를 하나의 DataFrame으로 통합."""
    frames = []
    for class_id, filepath in CLASS_FILES.items():
        if not filepath.exists():
            print(f"  [경고] {filepath} 없음 — 건너뜀")
            continue
        df = pd.read_excel(filepath)
        df["class_id"] = class_id
        df["student_id"] = class_id + "_" + df["이름"].astype(str)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("데이터 파일이 없습니다. data/ 폴더를 확인하세요.")

    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={
        "전공": "major", "프로젝트주제": "project",
        "에필로그": "epilogue", "활용": "application"
    })
    result["epi_len"] = result["epilogue"].str.len()
    result["app_len"] = result["application"].str.len()
    result["total_len"] = result["epi_len"] + result["app_len"]

    print(f"[데이터] {len(result)}명, {result['class_id'].nunique()}개 반 로드 완료")
    return result


# ═══════════════════════════════════════════════════════════
# 1단계: LLM 자유 추출
# ═══════════════════════════════════════════════════════════
def step1_llm_extract(df: pd.DataFrame) -> dict:
    """각 학생의 에필로그+활용에서 엔티티/관계를 LLM으로 자유 추출."""
    print(f"\n{'='*60}")
    print("1단계: LLM 자유 추출")
    print(f"{'='*60}")

    try:
        import anthropic
        client = anthropic.Anthropic()
        print("  Anthropic API 연결 성공")
    except Exception as e:
        print(f"  [경고] API 미연결: {e}")
        return {row["student_id"]: {"entities": [], "relations": []} for _, row in df.iterrows()}

    all_extractions = {}
    for idx, row in df.iterrows():
        sid = row["student_id"]
        user_prompt = LLM_USER_PROMPT_TEMPLATE.format(
            major=row["major"], project=row["project"],
            epilogue=row["epilogue"], application=row["application"]
        )
        try:
            response = client.messages.create(
                model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE, system=LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            raw_text = response.content[0].text.strip()
            raw_text = re.sub(r"^```json\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            extraction = json.loads(raw_text)
            all_extractions[sid] = extraction
            print(f"  {sid}: 엔티티 {len(extraction.get('entities',[]))}개, 관계 {len(extraction.get('relations',[]))}개")
        except json.JSONDecodeError as e:
            print(f"  [오류] {sid}: JSON 파싱 실패 — {e}")
            all_extractions[sid] = {"entities": [], "relations": []}
        except Exception as e:
            print(f"  [오류] {sid}: API 호출 실패 — {e}")
            all_extractions[sid] = {"entities": [], "relations": []}

    success = sum(1 for v in all_extractions.values() if v["entities"])
    print(f"\n  추출 완료: {success}/{len(df)}명")

    out_path = OUTPUT_DIR / "step1_extractions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_extractions, f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_path}")
    return all_extractions


def step1_load_existing() -> dict:
    """이전 추출 결과가 있으면 로드."""
    path = OUTPUT_DIR / "step1_extractions.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[1단계] 기존 추출 결과 로드: {len(data)}명")
        return data
    return None


# ═══════════════════════════════════════════════════════════
# 2단계: 엔티티 해상도 (2a: type 정규화 + 2b: 엔티티 병합)
#   2a: EDC(Zhang & Soh, 2024) self-canonicalization 원리
#   2b: Graphusion(Yang et al., 2024) knowledge fusion 원리
# ═══════════════════════════════════════════════════════════

TYPE_CANON_SYSTEM_PROMPT = """당신은 텍스트 분석 전문가입니다.
아래는 학습경험 요소에서 추출된 유형명(type) 목록입니다.
동일한 개념인데 표현만 다른 유형명을 하나의 대표 유형명으로 통일해 주세요.

규칙:
1. 띄어쓰기, 밑줄 등 표기만 다른 것은 반드시 통일하세요.
   예: "학습성과", "학습 성과", "학습_성과" → "학습성과"
2. 의미적으로 동일하거나 매우 유사한 유형명도 통일하세요.
   예: "학습성과", "학습효과", "학습결과" → "학습성과"
   예: "전공적용", "전공연계", "전공활용" → "전공적용"
3. 의미가 명확히 다르면 분리하세요.
4. 반드시 아래 JSON 형식으로만 응답하세요.

출력 형식:
{
  "type_groups": [
    {"canonical": "대표유형명", "members": ["원본1", "원본2"]}
  ]
}"""

TYPE_CANON_USER_TEMPLATE = """아래는 {n_types}개의 유형명입니다. 동일/유사한 것을 그룹핑하고 대표 유형명을 부여해 주세요.

{type_list}"""

MERGE_SYSTEM_PROMPT = """당신은 텍스트 분석 전문가입니다.
아래는 여러 학생의 성찰 텍스트에서 추출된 학습경험 요소(엔티티) 목록입니다.
의미적으로 동일하거나 매우 유사한 엔티티를 그룹핑해 주세요.

규칙:
1. 표현이 다르더라도 의미가 같으면 같은 그룹으로 묶으세요.
   예: "코딩에 대한 자신감 생성"과 "코딩 실력에 대한 자신감이 생김" → 같은 그룹
2. 의미가 다르면 다른 그룹으로 분리하세요.
3. 각 그룹에 대표 이름을 부여하세요.
4. 반드시 아래 JSON 형식으로만 응답하세요.

출력 형식:
{
  "groups": [
    {"group_id": 0, "representative_name": "대표 이름", "entity_indices": [0, 3, 7]}
  ]
}"""

MERGE_USER_PROMPT_TEMPLATE = """아래는 {n_entities}개의 학습경험 요소입니다. 의미적으로 동일하거나 유사한 것을 그룹핑해 주세요.

{entity_list}"""


def _normalize_type_surface(extractions: dict, all_entities: list):
    """표기 후처리: 띄어쓰기/밑줄을 제거한 형태로 모든 type을 통일.
    LLM이 놓친 표기 변형을 잡아준다.
    "학습 성과" → "학습성과", "학습_활동" → "학습활동"
    엔티티 type과 관계 type 모두 정규화.
    """
    # --- 엔티티 type 정규화 ---
    all_types = [e["type"] for e in all_entities]
    norm_groups = {}
    for t in set(all_types):
        norm_key = re.sub(r"[\s_\-]", "", t)
        if norm_key not in norm_groups:
            norm_groups[norm_key] = []
        norm_groups[norm_key].append(t)

    surface_map = {}
    for norm_key, variants in norm_groups.items():
        canonical = norm_key  # 띄어쓰기/밑줄 제거된 형태가 대표형
        for v in variants:
            if v != canonical:
                surface_map[v] = canonical

    if surface_map:
        for ent in all_entities:
            if ent["type"] in surface_map:
                ent["type"] = surface_map[ent["type"]]
        for sid, ext in extractions.items():
            for ent in ext.get("entities", []):
                if ent.get("type", "") in surface_map:
                    ent["type"] = surface_map[ent["type"]]
        print(f"    엔티티 표기 후처리: {len(surface_map)}개 변형 통일")

    # --- 관계 type 정규화 ---
    rel_surface_map = {}
    all_rel_types = set()
    for sid, ext in extractions.items():
        for rel in ext.get("relations", []):
            all_rel_types.add(rel.get("type", ""))

    rel_norm_groups = {}
    for t in all_rel_types:
        norm_key = re.sub(r"[\s_\-]", "", t)
        if norm_key not in rel_norm_groups:
            rel_norm_groups[norm_key] = []
        rel_norm_groups[norm_key].append(t)

    for norm_key, variants in rel_norm_groups.items():
        canonical = norm_key
        for v in variants:
            if v != canonical:
                rel_surface_map[v] = canonical

    if rel_surface_map:
        for sid, ext in extractions.items():
            for rel in ext.get("relations", []):
                if rel.get("type", "") in rel_surface_map:
                    rel["type"] = rel_surface_map[rel["type"]]
        print(f"    관계 표기 후처리: {len(rel_surface_map)}개 변형 통일")


def step2_merge_entities(extractions: dict, skip_merge: bool = False) -> dict:
    """LLM 기반 엔티티 해상도 (2a + 2b 분리 실행).

    2a: type 정규화 (LLM 의미 정규화 + 표기 후처리)
    2b: 엔티티 병합 (반별 LLM fusion)

    캐시가 있으면 재사용하되, extractions에 type 정규화는 항상 적용.
    """
    print(f"\n{'='*60}")
    print("2단계: 엔티티 해상도 (2a: type 정규화 + 2b: 엔티티 병합)")
    print(f"{'='*60}")

    # 모든 엔티티 수집
    all_entities = []
    for sid, ext in extractions.items():
        for ent in ext.get("entities", []):
            all_entities.append({
                "student_id": sid, "entity_id": ent["id"],
                "text": ent.get("text", ""), "type": ent.get("type", ""),
                "source": ent.get("source", ""), "key": f"{sid}:{ent['id']}",
            })

    if len(all_entities) < 2:
        print("  엔티티 부족 — 병합 생략")
        node_map = {e["key"]: e["key"] for e in all_entities}
        return {"node_map": node_map, "shared_nodes": {}, "extractions": extractions}

    # ══════════════════════════════════════
    # 2a: type 정규화 (항상 실행 — 캐시 여부와 무관)
    # ══════════════════════════════════════
    print(f"\n  [2a] type 정규화...")
    unique_types = sorted(set(e["type"] for e in all_entities))
    print(f"    정규화 전 고유 type: {len(unique_types)}개")

    # LLM 의미 정규화
    type_map = {t: t for t in unique_types}
    try:
        import anthropic
        client = anthropic.Anthropic()

        if len(unique_types) > 1:
            type_list_text = "\n".join(f"[{i}] {t}" for i, t in enumerate(unique_types))
            response = client.messages.create(
                model=LLM_MODEL, max_tokens=4000, temperature=LLM_TEMPERATURE,
                system=TYPE_CANON_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": TYPE_CANON_USER_TEMPLATE.format(
                    n_types=len(unique_types), type_list=type_list_text)}]
            )
            raw_text = response.content[0].text.strip()
            raw_text = re.sub(r"^```json\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            canon_result = json.loads(raw_text)

            for group in canon_result.get("type_groups", []):
                canonical = group.get("canonical", "")
                for member in group.get("members", []):
                    if member in type_map:
                        type_map[member] = canonical

            n_merged = sum(1 for k, v in type_map.items() if k != v)
            print(f"    LLM 의미 정규화: {n_merged}개 통합")
    except Exception as e:
        print(f"    [경고] LLM type 정규화 실패: {e} — 표기 후처리만 적용")

    # type_map 적용
    for ent in all_entities:
        ent["type"] = type_map.get(ent["type"], ent["type"])
    for sid, ext in extractions.items():
        for ent in ext.get("entities", []):
            ent["type"] = type_map.get(ent.get("type", ""), ent.get("type", ""))

    # 표기 후처리 (띄어쓰기/밑줄 제거)
    _normalize_type_surface(extractions, all_entities)

    canonical_unique = len(set(e["type"] for e in all_entities))
    print(f"    최종 고유 type: {canonical_unique}개")

    # 정규화된 extractions를 step1에 덮어쓰기 저장
    step1_path = OUTPUT_DIR / "step1_extractions.json"
    with open(step1_path, "w", encoding="utf-8") as f:
        json.dump(extractions, f, ensure_ascii=False, indent=2)
    print(f"    정규화 결과 저장: {step1_path}")

    # ══════════════════════════════════════
    # 2c: type 상위 범주화 (반별 독립 도출)
    # ══════════════════════════════════════
    print(f"\n  [2c] type 상위 범주화 (반별 독립)...")
    type_category_map = {}  # (class_id, type) → 상위 범주
    class_categories = {}   # {class_id: [{"category": ..., "members": [...]}, ...]}

    category_cache_path = OUTPUT_DIR / "type_categories.json"
    if category_cache_path.exists():
        try:
            with open(category_cache_path, "r", encoding="utf-8") as f:
                cached_cats = json.load(f)
            class_categories = cached_cats
            for cid, cats in class_categories.items():
                for cat in cats:
                    for m in cat.get("members", []):
                        type_category_map[(cid, m)] = cat["category"]
            print(f"    기존 범주화 로드: {list(class_categories.keys())}")
        except:
            pass

    if not class_categories:
        try:
            import anthropic
            client_cat = anthropic.Anthropic()

            CAT_SYSTEM = """당신은 교육학 전문가입니다.
아래는 한 반의 PBL 학습자 성찰 텍스트에서 귀납적으로 추출된 학습경험 요소의 유형명 목록입니다.
이 유형들을 5~8개의 상위 범주로 그룹핑해 주세요.

규칙:
1. 이 반의 데이터에서 나타나는 특성을 반영한 범주명을 부여하세요.
2. 모든 유형이 반드시 하나의 상위 범주에 포함되어야 합니다.
3. 상위 범주 간에 겹치지 않아야 합니다.
4. 반드시 아래 JSON 형식으로만 응답하세요.

출력 형식:
{
  "categories": [
    {
      "category": "상위 범주명",
      "description": "이 범주의 의미 설명 (1~2문장)",
      "members": ["유형1", "유형2", "유형3"]
    }
  ]
}"""

            # 반별 엔티티 그룹
            class_ents = defaultdict(list)
            for ent in all_entities:
                cid = ent["student_id"].split("_")[0]
                class_ents[cid].append(ent)

            for cid in sorted(class_ents.keys()):
                ents_in_class = class_ents[cid]
                from collections import Counter as _Ctr
                _tc = _Ctr(e["type"] for e in ents_in_class)
                class_types = sorted(_tc.keys(), key=lambda x: -_tc[x])
                type_list_text = "\n".join(f"[{i}] {t} ({_tc[t]}회)" for i, t in enumerate(class_types))

                response = client_cat.messages.create(
                    model=LLM_MODEL, max_tokens=4000, temperature=LLM_TEMPERATURE,
                    system=CAT_SYSTEM,
                    messages=[{"role": "user", "content":
                        f"아래는 반 {cid}의 {len(class_types)}개 학습경험 요소 유형입니다. 5~8개 상위 범주로 그룹핑해 주세요.\n\n{type_list_text}"}]
                )
                raw_text = response.content[0].text.strip()
                raw_text = re.sub(r"^```json\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
                cat_result = json.loads(raw_text)

                cats = cat_result.get("categories", [])
                class_categories[cid] = cats
                for cat in cats:
                    for m in cat.get("members", []):
                        type_category_map[(cid, m)] = cat["category"]

                # 매핑 누락 type 확인 및 2차 호출
                mapped_types = set()
                for cat in cats:
                    mapped_types.update(cat.get("members", []))
                unmapped = [t for t in class_types if t not in mapped_types]

                if unmapped and len(unmapped) <= len(class_types) * 0.7:
                    print(f"      미매핑 {len(unmapped)}개 → 2차 분류 시도")
                    unmapped_text = "\n".join(f"[{i}] {t}" for i, t in enumerate(unmapped))
                    existing_cats = ", ".join(cat["category"] for cat in cats)
                    retry_prompt = (
                        f"아래는 이전 분류에서 누락된 {len(unmapped)}개의 유형입니다.\n"
                        f"기존 범주: {existing_cats}\n"
                        f"이 유형들을 기존 범주 중 하나에 배정하세요. 어디에도 맞지 않으면 '기타'로 배정하세요.\n"
                        f"반드시 JSON 형식으로 응답하세요: {{\"assignments\": [{{\"type\": \"유형명\", \"category\": \"범주명\"}}]}}\n\n"
                        f"{unmapped_text}"
                    )
                    try:
                        retry_resp = client_cat.messages.create(
                            model=LLM_MODEL, max_tokens=4000, temperature=LLM_TEMPERATURE,
                            system="당신은 교육학 전문가입니다. 학습경험 요소를 기존 범주에 분류합니다.",
                            messages=[{"role": "user", "content": retry_prompt}]
                        )
                        retry_text = retry_resp.content[0].text.strip()
                        retry_text = re.sub(r"^```json\s*", "", retry_text)
                        retry_text = re.sub(r"\s*```$", "", retry_text)
                        retry_result = json.loads(retry_text)

                        n_assigned = 0
                        for assignment in retry_result.get("assignments", []):
                            t = assignment.get("type", "")
                            c = assignment.get("category", "기타")
                            type_category_map[(cid, t)] = c
                            # 기존 cats의 members에도 추가
                            for cat in cats:
                                if cat["category"] == c:
                                    if t not in cat.get("members", []):
                                        cat["members"].append(t)
                                    break
                            n_assigned += 1
                        print(f"      2차 분류: {n_assigned}개 추가 배정")
                    except Exception as e2:
                        print(f"      2차 분류 실패: {e2}")

                print(f"    반 {cid}: {len(cats)}개 상위 범주 도출")
                for cat in cats:
                    n_m = len(cat.get("members", []))
                    n_e = sum(_tc.get(m, 0) for m in cat.get("members", []))
                    print(f"      {cat['category']}: {n_m}개 유형, {n_e}개 엔티티")

            # 캐시 저장
            with open(category_cache_path, "w", encoding="utf-8") as f:
                json.dump(class_categories, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"    [경고] 상위 범주화 실패: {e}")

    # 엔티티에 상위 범주 속성 추가
    for ent in all_entities:
        cid = ent["student_id"].split("_")[0]
        ent["category"] = type_category_map.get((cid, ent["type"]), "기타")
    for sid, ext in extractions.items():
        cid = sid.split("_")[0]
        for ent in ext.get("entities", []):
            ent["category"] = type_category_map.get((cid, ent.get("type", "")), "기타")

    # category 부여 후 step1 재저장
    step1_path = OUTPUT_DIR / "step1_extractions.json"
    with open(step1_path, "w", encoding="utf-8") as f:
        json.dump(extractions, f, ensure_ascii=False, indent=2)
    n_categorized = sum(1 for ext in extractions.values() for e in ext.get("entities",[]) if e.get("category","기타") != "기타")
    print(f"    category 부여 완료: {n_categorized}/{len(all_entities)}개 정상 매핑")
    print(f"    step1 재저장: {step1_path}")

    # ══════════════════════════════════════
    # 캐시 확인 (2b 결과만 캐시)
    # ══════════════════════════════════════
    merge_cache_path = OUTPUT_DIR / "step2_merge_cache.json"

    # skip_merge 모드: 2b 건너뛰고 1:1 매핑
    if skip_merge:
        print(f"\n  [2b] 병합 건너뜀 (--skip-merge)")
        node_map = {e["key"]: e["key"] for e in all_entities}
        shared_nodes = {}
        for e in all_entities:
            shared_nodes[e["key"]] = {
                "type": e["type"], "text": e["text"],
                "students": [e["student_id"]], "n_students": 1,
                "all_texts": [e["text"]],
                "sources": [e.get("source", "")],
            }
        cache_data = {"n_entities": len(all_entities), "node_map": node_map, "shared_nodes": shared_nodes}
        with open(merge_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        return {"node_map": node_map, "shared_nodes": shared_nodes, "extractions": extractions}

    if merge_cache_path.exists():
        try:
            with open(merge_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("n_entities") == len(all_entities):
                print(f"\n  [2b] 기존 병합 결과 로드 (공유 노드 {len(cached['shared_nodes'])}개)")
                print(f"  → 재병합하려면 outputs/step2_merge_cache.json 삭제")
                return {
                    "node_map": cached["node_map"],
                    "shared_nodes": cached["shared_nodes"],
                    "extractions": extractions,
                }
        except:
            pass

    # ══════════════════════════════════════
    # 2b: 엔티티 병합 (반별 LLM 호출 — 청크 단위)
    # ══════════════════════════════════════
    print(f"\n  [2b] 엔티티 병합...")
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"  [경고] API 미연결: {e} — 1:1 매핑")
        node_map = {e["key"]: e["key"] for e in all_entities}
        return {"node_map": node_map, "shared_nodes": {}, "extractions": extractions}

    node_map = {}
    shared_nodes = {}
    shared_id_counter = 0

    class_entities = defaultdict(list)
    for i, ent in enumerate(all_entities):
        class_id = ent["student_id"].split("_")[0]
        class_entities[class_id].append((i, ent))

    CHUNK_SIZE = 50  # LLM 응답 안정성을 위해 50개씩 처리

    for class_id in sorted(class_entities.keys()):
        ents = class_entities[class_id]
        print(f"\n    반 {class_id}: {len(ents)}개 엔티티 병합 중...")

        # 청크 분할
        chunks = [ents[i:i+CHUNK_SIZE] for i in range(0, len(ents), CHUNK_SIZE)]
        print(f"      → {len(chunks)}개 청크로 분할 (각 {CHUNK_SIZE}개)")

        class_grouped_indices = set()

        for chunk_idx, chunk in enumerate(chunks):
            entity_lines = []
            for idx, (global_idx, ent) in enumerate(chunk):
                entity_lines.append(f"[{idx}] type=\"{ent['type']}\" text=\"{ent['text']}\"")

            user_prompt = MERGE_USER_PROMPT_TEMPLATE.format(
                n_entities=len(chunk), entity_list="\n".join(entity_lines))

            # 최대 2회 재시도
            merge_success = False
            for attempt in range(2):
                try:
                    response = client.messages.create(
                        model=LLM_MODEL, max_tokens=8000, temperature=LLM_TEMPERATURE,
                        system=MERGE_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    raw_text = response.content[0].text.strip()
                    raw_text = re.sub(r"^```json\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)

                    # 불완전 JSON 복구 시도
                    try:
                        merge_result = json.loads(raw_text)
                    except json.JSONDecodeError:
                        # 잘린 JSON 복구: 마지막 완전한 } 찾기
                        last_brace = raw_text.rfind("}")
                        if last_brace > 0:
                            truncated = raw_text[:last_brace+1]
                            # 닫히지 않은 배열/객체 보완
                            open_brackets = truncated.count("[") - truncated.count("]")
                            open_braces = truncated.count("{") - truncated.count("}")
                            truncated += "]" * open_brackets + "}" * open_braces
                            merge_result = json.loads(truncated)
                        else:
                            raise

                    groups = merge_result.get("groups", [])
                    chunk_grouped = set()

                    for group in groups:
                        group_indices = group.get("entity_indices", [])
                        rep_name = group.get("representative_name", "")
                        members = []
                        for local_idx in group_indices:
                            if 0 <= local_idx < len(chunk):
                                global_idx, ent = chunk[local_idx]
                                members.append(ent)
                                chunk_grouped.add(local_idx)
                        if not members:
                            continue

                        student_set = set(m["student_id"] for m in members)
                        shared_id = f"SN_{shared_id_counter}"
                        shared_id_counter += 1
                        shared_nodes[shared_id] = {
                            "type": members[0]["type"],
                            "text": rep_name,
                            "students": list(student_set),
                            "n_students": len(student_set),
                            "all_texts": [m["text"] for m in members],
                            "sources": list(set(m.get("source", "") for m in members)),
                        }
                        for m in members:
                            node_map[m["key"]] = shared_id

                    # 미그룹 엔티티
                    for local_idx, (global_idx, ent) in enumerate(chunk):
                        if local_idx not in chunk_grouped:
                            shared_id = f"SN_{shared_id_counter}"
                            shared_id_counter += 1
                            shared_nodes[shared_id] = {
                                "type": ent["type"], "text": ent["text"],
                                "students": [ent["student_id"]], "n_students": 1,
                                "all_texts": [ent["text"]],
                                "sources": [ent.get("source", "")],
                            }
                            node_map[ent["key"]] = shared_id

                    print(f"      청크 {chunk_idx+1}/{len(chunks)}: {len(groups)}개 그룹 도출")
                    merge_success = True
                    break  # 성공하면 재시도 루프 탈출

                except Exception as e:
                    if attempt == 0:
                        print(f"      청크 {chunk_idx+1} 시도 {attempt+1} 실패: {e} → 재시도...")
                    else:
                        print(f"      청크 {chunk_idx+1} 시도 {attempt+1} 실패: {e} → 1:1 매핑")

            # 2회 재시도 모두 실패 시 1:1 매핑
            if not merge_success:
                for local_idx, (global_idx, ent) in enumerate(chunk):
                    if ent["key"] not in node_map:
                        shared_id = f"SN_{shared_id_counter}"
                        shared_id_counter += 1
                        shared_nodes[shared_id] = {
                            "type": ent["type"], "text": ent["text"],
                            "students": [ent["student_id"]], "n_students": 1,
                            "all_texts": [ent["text"]],
                            "sources": [ent.get("source", "")],
                        }
                        node_map[ent["key"]] = shared_id

    # 통계
    multi_student = sum(1 for sn in shared_nodes.values() if sn["n_students"] > 1)
    print(f"\n  최종 공유 노드: {len(shared_nodes)}개 (압축률: {len(all_entities)/max(len(shared_nodes),1):.1f}x)")
    print(f"  다수 학생 공유 노드: {multi_student}개")

    top_shared = sorted(shared_nodes.items(), key=lambda x: x[1]["n_students"], reverse=True)[:10]
    for sid, info in top_shared:
        if info["n_students"] > 1:
            print(f"    {sid}: [{info['type']}] \"{info['text']}\" — {info['n_students']}명")

    # 캐시 저장
    cache_data = {"n_entities": len(all_entities), "node_map": node_map, "shared_nodes": shared_nodes}
    with open(merge_cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    print(f"  캐시 저장: {merge_cache_path}")

    return {"node_map": node_map, "shared_nodes": shared_nodes, "extractions": extractions}


# ═══════════════════════════════════════════════════════════
# 3단계: 그래프 구성 + 커뮤니티 탐지
# ═══════════════════════════════════════════════════════════
def step3_build_graph(df: pd.DataFrame, extractions: dict,
                      merge_result: dict = None) -> dict:
    """공유 노드 기반 반별 그래프 구성 및 Louvain 커뮤니티 탐지."""
    print(f"\n{'='*60}")
    print("3단계: 그래프 구성 + 커뮤니티 탐지")
    print(f"{'='*60}")

    node_map = merge_result.get("node_map", {}) if merge_result else {}
    shared_nodes = merge_result.get("shared_nodes", {}) if merge_result else {}
    results = {}

    for class_id in sorted(df["class_id"].unique()):
        class_df = df[df["class_id"] == class_id]
        G = nx.Graph()

        for _, row in class_df.iterrows():
            sid = row["student_id"]
            ext = extractions.get(sid, {"entities": [], "relations": []})

            student_node_map = {}
            for ent in ext.get("entities", []):
                original_key = f"{sid}:{ent['id']}"
                shared_id = node_map.get(original_key, original_key)
                student_node_map[ent["id"]] = shared_id
                sn_info = shared_nodes.get(shared_id, {})

                if not G.has_node(shared_id):
                    G.add_node(shared_id,
                        type=sn_info.get("type", ent.get("type", "")),
                        text=sn_info.get("text", ent.get("text", "")),
                        sources=sn_info.get("sources", [ent.get("source", "")]),
                        students=[sid], n_students=1)
                else:
                    students = G.nodes[shared_id].get("students", [])
                    if sid not in students:
                        students.append(sid)
                        G.nodes[shared_id]["students"] = students
                        G.nodes[shared_id]["n_students"] = len(students)

            for rel in ext.get("relations", []):
                src = student_node_map.get(rel["source"])
                tgt = student_node_map.get(rel["target"])
                if src and tgt and G.has_node(src) and G.has_node(tgt) and src != tgt:
                    if G.has_edge(src, tgt):
                        G[src][tgt]["weight"] = G[src][tgt].get("weight", 1) + 1
                    else:
                        G.add_edge(src, tgt, type=rel.get("type", ""), weight=1)

        multi = sum(1 for n in G.nodes if G.nodes[n].get("n_students", 1) > 1)
        print(f"  반 {class_id}: 노드 {G.number_of_nodes()}개 (공유 {multi}개), 엣지 {G.number_of_edges()}개")

        # 커뮤니티 탐지
        communities = {}
        community_labels = {}
        try:
            import community as community_louvain
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                partition = community_louvain.best_partition(G, resolution=COMMUNITY_RESOLUTION, random_state=42)
                comm_groups = defaultdict(list)
                for node, comm_id in partition.items():
                    comm_groups[comm_id].append(node)

                for comm_id, nodes in comm_groups.items():
                    types = [G.nodes[n].get("type", "") for n in nodes]
                    type_counts = Counter(types).most_common(3)
                    label = " / ".join([t for t, _ in type_counts if t])
                    communities[comm_id] = nodes
                    community_labels[comm_id] = label if label else f"범주_{comm_id}"

                modularity = community_louvain.modularity(partition, G)
                print(f"         커뮤니티: {len(communities)}개 (modularity={modularity:.3f})")
        except ImportError:
            print("  [경고] python-louvain 미설치")

        results[class_id] = {
            "graph": G, "communities": communities,
            "community_labels": community_labels,
            "node_map": {k: v for k, v in node_map.items() if k.startswith(class_id + "_")},
        }

    return results


# ═══════════════════════════════════════════════════════════
# 4단계: 구조적 지표 산출
# ═══════════════════════════════════════════════════════════
def step4_compute_metrics(df: pd.DataFrame, extractions: dict, graph_results: dict) -> pd.DataFrame:
    """비율 기반 구조적 지표 산출 + 길이독립성/VIF 검증."""
    print(f"\n{'='*60}")
    print("4단계: 구조적 지표 산출")
    print(f"{'='*60}")

    records = []
    for _, row in df.iterrows():
        sid = row["student_id"]
        class_id = row["class_id"]
        ext = extractions.get(sid, {"entities": [], "relations": []})
        entities = ext.get("entities", [])
        relations = ext.get("relations", [])

        n_entities = len(entities)
        n_relations = len(relations)
        epi_entities = [e for e in entities if e.get("source", "").startswith("에필로그")]
        app_entities = [e for e in entities if e.get("source", "").startswith("활용")]
        epi_ids = {e["id"] for e in epi_entities}
        app_ids = {e["id"] for e in app_entities}

        cross_edges = sum(1 for r in relations
                         if (r["source"] in epi_ids and r["target"] in app_ids)
                         or (r["source"] in app_ids and r["target"] in epi_ids))

        relation_entity_ratio = n_relations / n_entities if n_entities > 0 else 0
        cross_ratio = cross_edges / n_relations if n_relations > 0 else 0

        # connectivity
        gr_data = graph_results.get(class_id, {})
        G = gr_data.get("graph", nx.Graph())
        local_node_map = gr_data.get("node_map", {})
        connectivity = 0
        if n_entities > 0:
            student_shared = set()
            for ent in entities:
                shared_id = local_node_map.get(f"{sid}:{ent['id']}", f"{sid}:{ent['id']}")
                if G.has_node(shared_id):
                    student_shared.add(shared_id)
            if student_shared:
                sub = G.subgraph(student_shared)
                if sub.number_of_nodes() > 0:
                    comps = list(nx.connected_components(sub))
                    connectivity = max(len(c) for c in comps) / sub.number_of_nodes()

        # coverage
        communities = gr_data.get("communities", {})
        coverage = 0
        if communities and n_entities > 0:
            student_shared_set = set()
            for ent in entities:
                student_shared_set.add(local_node_map.get(f"{sid}:{ent['id']}", f"{sid}:{ent['id']}"))
            covered = sum(1 for comm_nodes in communities.values() if student_shared_set & set(comm_nodes))
            coverage = covered / len(communities)

        # community_cross_ratio
        community_cross_ratio = 0
        if communities and n_relations > 0:
            node_to_comm = {}
            for comm_id, comm_nodes in communities.items():
                for n in comm_nodes:
                    node_to_comm[n] = comm_id
            cross_comm = 0
            for rel in relations:
                src_shared = local_node_map.get(f"{sid}:{rel['source']}", f"{sid}:{rel['source']}")
                tgt_shared = local_node_map.get(f"{sid}:{rel['target']}", f"{sid}:{rel['target']}")
                sc = node_to_comm.get(src_shared)
                tc = node_to_comm.get(tgt_shared)
                if sc is not None and tc is not None and sc != tc:
                    cross_comm += 1
            community_cross_ratio = cross_comm / n_relations

        records.append({
            "student_id": sid, "class_id": class_id, "major": row["major"],
            "epi_len": row["epi_len"], "app_len": row["app_len"], "total_len": row["total_len"],
            "n_entities": n_entities, "n_relations": n_relations,
            "n_epi_entities": len(epi_entities), "n_app_entities": len(app_entities),
            "cross_edges": cross_edges,
            "relation_entity_ratio": round(relation_entity_ratio, 4),
            "cross_ratio": round(cross_ratio, 4),
            "connectivity": round(connectivity, 4),
            "coverage": round(coverage, 4),
            "community_cross_ratio": round(community_cross_ratio, 4),
        })

    metrics_df = pd.DataFrame(records)

    # 길이 독립성 검증
    ratio_cols = ["relation_entity_ratio", "cross_ratio", "connectivity", "coverage", "community_cross_ratio"]
    print(f"\n  길이 독립성 검증 (|ρ| < {LENGTH_CORRELATION_THRESHOLD}):")
    for col in ratio_cols:
        if metrics_df[col].std() == 0:
            print(f"    {col}: 분산 0 — 제외")
            continue
        rho, p = stats.spearmanr(metrics_df["total_len"], metrics_df[col])
        status = "✓" if abs(rho) < LENGTH_CORRELATION_THRESHOLD else "✗"
        print(f"    {col}: ρ={rho:.3f} (p={p:.3f}) {status}")

    out_path = OUTPUT_DIR / "step4_metrics.csv"
    metrics_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {out_path}")
    return metrics_df


# ═══════════════════════════════════════════════════════════
# 5단계: 반 내 상대 위치 판정 + 보고서
# ═══════════════════════════════════════════════════════════
def step5_position_and_report(df, metrics_df, extractions, graph_results):
    """반 내 백분위 산출 및 보고서 생성."""
    print(f"\n{'='*60}")
    print("5단계: 반 내 상대 위치 판정")
    print(f"{'='*60}")

    ratio_cols = ["relation_entity_ratio", "cross_ratio", "connectivity", "coverage", "community_cross_ratio"]

    for col in ratio_cols:
        pct_col = f"{col}_pct"
        metrics_df[pct_col] = 0.0
        for class_id in metrics_df["class_id"].unique():
            mask = metrics_df["class_id"] == class_id
            values = metrics_df.loc[mask, col]
            metrics_df.loc[mask, pct_col] = values.rank(pct=True) * 100 if values.std() > 0 else 50.0

    # 보고서 생성
    R = []  # report lines
    R.append("# PBL 학습경험 평가 보고서\n")
    R.append(f"분석 대상: {len(metrics_df)}명, {metrics_df['class_id'].nunique()}개 반\n")

    for class_id in sorted(metrics_df["class_id"].unique()):
        cd = metrics_df[metrics_df["class_id"] == class_id]
        gr = graph_results.get(class_id, {})
        communities = gr.get("communities", {})
        labels = gr.get("community_labels", {})

        R.append(f"\n---\n## 반 {class_id} ({len(cd)}명)\n")

        if labels:
            R.append(f"### 도출된 학습경험 범주 ({len(labels)}개)\n")
            for cid, label in labels.items():
                R.append(f"  - 범주 {cid}: {label} ({len(communities.get(cid, []))}개 노드)")

        R.append(f"\n### 지표 기술통계\n")
        for col in ratio_cols:
            v = cd[col]
            R.append(f"  - {col}: 평균={v.mean():.3f}, SD={v.std():.3f}, 범위={v.min():.3f}~{v.max():.3f}")

        # 반별 해석
        R.append(f"\n### 반 {class_id} 해석\n")
        rer_mean = cd["relation_entity_ratio"].mean()
        cr_mean = cd["cross_ratio"].mean()
        ccr_mean = cd["community_cross_ratio"].mean()
        cr_zero = len(cd[cd["cross_ratio"] == 0])
        cr_high = len(cd[cd["cross_ratio"] > cr_mean])
        cr_std = cd["cross_ratio"].std()

        if rer_mean >= 0.85:
            R.append(f"  - **구조적 밀도**: 높음 (평균 {rer_mean:.3f})")
        elif rer_mean >= 0.75:
            R.append(f"  - **구조적 밀도**: 보통 (평균 {rer_mean:.3f})")
        else:
            R.append(f"  - **구조적 밀도**: 낮음 (평균 {rer_mean:.3f})")

        if cr_zero > 0:
            R.append(f"  - **전이 부재 학생**: {cr_zero}명")
        R.append(f"  - **전이 활발 학생**: {cr_high}명")
        R.append(f"  - **전이 편차**: {'큼' if cr_std > 0.10 else '작음'} (SD={cr_std:.3f})")
        R.append(f"  - **범주 간 통합**: {'활발' if ccr_mean >= 0.30 else '제한적'} (평균 {ccr_mean:.3f})")

        # 개인별 프로파일
        R.append(f"\n### 개인별 프로파일\n")
        rer_med = cd["relation_entity_ratio"].median()
        cr_med = cd["cross_ratio"].median()

        for _, row in cd.iterrows():
            sid = row["student_id"]
            ext = extractions.get(sid, {"entities": [], "relations": []})
            entities = ext.get("entities", [])
            relations = ext.get("relations", [])

            R.append(f"\n#### {sid} ({row['major']})")
            R.append(f"  - 엔티티: {row['n_entities']}개 (에필로그 {row['n_epi_entities']}, 활용 {row['n_app_entities']})")
            R.append(f"  - 관계: {row['n_relations']}개 (교차 연결 {row['cross_edges']}개)")

            for col in ratio_cols:
                R.append(f"  - {col}: {row[col]:.3f} (반 내 {row[f'{col}_pct']:.0f}%)")

            epi_ents = [e for e in entities if e.get("source", "").startswith("에필로그")]
            app_ents = [e for e in entities if e.get("source", "").startswith("활용")]
            if epi_ents:
                R.append(f"  - [에필로그 요소] {', '.join(e.get('text','') for e in epi_ents)}")
            if app_ents:
                R.append(f"  - [활용 요소] {', '.join(e.get('text','') for e in app_ents)}")

            epi_ids = {e["id"] for e in epi_ents}
            app_ids = {e["id"] for e in app_ents}
            id_to_text = {e["id"]: e.get("text", "") for e in entities}
            for rel in relations:
                if rel["source"] in epi_ids and rel["target"] in app_ids:
                    R.append(f"  - [전이 경로] \"{id_to_text.get(rel['source'],'')}\" →({rel.get('type','')})→ \"{id_to_text.get(rel['target'],'')}\"")

            # 진단 코멘트
            rer_val = row["relation_entity_ratio"]
            cr_val = row["cross_ratio"]
            R.append(f"\n  **[진단]**")
            if rer_val >= rer_med and cr_val >= cr_med:
                R.append(f"  - 포지션: 구조적 밀도 높음 + 전이 활발")
                R.append(f"  - 강점: 학습경험을 밀접하게 연결하면서 전공·일상 적용도 구체적으로 구상함.")
            elif rer_val < rer_med and cr_val >= cr_med:
                R.append(f"  - 포지션: 구조적 밀도 낮음 + 전이 활발")
                R.append(f"  - 개선 권고: 경험 요소 간 연결을 더 풍부하게 형성하면 성찰이 심화될 수 있음.")
            elif rer_val >= rer_med and cr_val < cr_med:
                R.append(f"  - 포지션: 구조적 밀도 높음 + 전이 부족")
                R.append(f"  - 개선 권고: 배운 것을 전공이나 일상에 어떻게 적용할 수 있는지 구체적으로 연결지어 보면 좋겠음.")
            else:
                R.append(f"  - 포지션: 구조적 밀도 낮음 + 전이 부족")
                R.append(f"  - 개선 권고: 경험 간 연결을 형성하고, 전공·일상 적용을 구체화해 보면 좋겠음.")

    # 길이 독립성
    R.append(f"\n---\n## 길이 독립성 검증\n")
    for col in ratio_cols:
        if metrics_df[col].std() > 0:
            rho, p = stats.spearmanr(metrics_df["total_len"], metrics_df[col])
            R.append(f"  - {col} vs 텍스트길이: ρ={rho:.3f} (p={p:.3f})")

    report_path = OUTPUT_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print(f"  보고서 저장: {report_path}")

    final_path = OUTPUT_DIR / "final_scores.csv"
    metrics_df.to_csv(final_path, index=False, encoding="utf-8-sig")
    print(f"  점수 저장: {final_path}")

    return metrics_df


# ═══════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PBL 학습경험 동적 평가 파이프라인")
    parser.add_argument("--fresh", action="store_true",
                       help="모든 캐시를 삭제하고 처음부터 실행 (step1 포함, API 비용 발생)")
    parser.add_argument("--refresh-cache", action="store_true",
                       help="step1은 보존하고 2단계 이후 캐시만 삭제 (API 비용 최소)")
    parser.add_argument("--skip-merge", action="store_true",
                       help="2b 엔티티 병합을 건너뜀 (병합 오류 시 사용)")
    args = parser.parse_args()

    print("PBL 학습경험 동적 평가 프레임워크")
    print("5단계 귀납적 파이프라인")
    print(f"{'='*60}\n")

    # 삭제 대상 결정
    if args.fresh or args.refresh_cache:
        cache_files = [
            OUTPUT_DIR / "step2_merge_cache.json",
            OUTPUT_DIR / "type_categories.json",
            OUTPUT_DIR / "step4_metrics.csv",
            OUTPUT_DIR / "final_scores.csv",
            OUTPUT_DIR / "evaluation_report.md",
        ]
        if args.fresh:
            cache_files.insert(0, OUTPUT_DIR / "step1_extractions.json")
            label = "모든 캐시"
        else:
            label = "2단계 이후 캐시 (step1 보존)"

        for cf in cache_files:
            if cf.exists():
                cf.unlink()
                print(f"  [삭제] {cf}")
        print(f"  → {label} 삭제 완료.\n")

    df = load_all_data()

    existing = step1_load_existing() if not args.fresh else None
    if existing and len(existing) >= len(df):
        extractions = existing
        print("  → 기존 추출 결과 사용")
    else:
        extractions = step1_llm_extract(df)

    merge_result = step2_merge_entities(extractions, skip_merge=args.skip_merge)
    extractions = merge_result["extractions"]

    graph_results = step3_build_graph(df, extractions, merge_result)
    metrics_df = step4_compute_metrics(df, extractions, graph_results)
    final_df = step5_position_and_report(df, metrics_df, extractions, graph_results)

    print(f"\n{'='*60}")
    print("파이프라인 완료")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
