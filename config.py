"""
config.py — 설정 파일 (완전 귀납적 접근)

핵심: ENTITY_TYPES, RELATION_TYPES 사전 정의 없음.
LLM이 텍스트에서 자유롭게 추출하고, 사후에 이론과 대조한다.
"""
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_FILES = {
    "A": DATA_DIR / "ClassA.xlsx",
    "B": DATA_DIR / "ClassB.xlsx",
    "C": DATA_DIR / "ClassC.xlsx",
    "D": DATA_DIR / "ClassD.xlsx",
}

LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 2000
LLM_COMBINE_SECTIONS = True

LLM_SYSTEM_PROMPT = """당신은 교육 연구자입니다.
학습자의 성찰 텍스트에서 학습경험과 관련된 요소(entity)와 그 관계(relation)를 추출합니다.

규칙:
1. 사전 정의된 유형을 사용하지 마세요. 텍스트에서 발견되는 요소를 있는 그대로 추출하세요.
2. 각 요소에 당신이 판단한 유형명을 자유롭게 부여하세요.
3. 요소 간 관계도 자유롭게 명명하세요.
4. 텍스트에 명시된 내용만 추출하세요. 추론하지 마세요.
5. 각 요소가 [에필로그]에서 나왔는지 [활용]에서 나왔는지 반드시 표시하세요.
6. 두 텍스트에 걸쳐 연결되는 요소가 있으면 그 관계도 추출하세요.
7. 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트를 포함하지 마세요.

출력 형식:
{
  "entities": [
    {"id": "E1", "type": "자유롭게 명명", "text": "원문에서 해당 부분 요약", "source": "에필로그 또는 활용"}
  ],
  "relations": [
    {"source": "E1", "target": "E2", "type": "자유롭게 명명"}
  ]
}"""

LLM_USER_PROMPT_TEMPLATE = """아래는 한 학생의 두 가지 성찰 텍스트입니다.

학생 정보:
- 전공: {major}
- 프로젝트 주제: {project}

[에필로그] PBL 프로젝트 종료 후 작성한 회고
\"\"\"{epilogue}\"\"\"

[활용] 학습 내용을 전공이나 일상에 적용한다면의 답변
\"\"\"{application}\"\"\"

위 두 텍스트에서 학습경험과 관련된 요소와 관계를 추출해 주세요."""

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MERGE_SIMILARITY_THRESHOLD = 0.55

COMMUNITY_ALGORITHM = "louvain"
COMMUNITY_RESOLUTION = 0.3

VIF_THRESHOLD = 5.0
LENGTH_CORRELATION_THRESHOLD = 0.3

STABILITY_TEST_N = 5
STABILITY_TEST_STUDENTS = 5
STABILITY_THRESHOLD = 0.95
