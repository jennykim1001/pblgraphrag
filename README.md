# PBL 학습경험 동적 평가 프레임워크 — 실행 가이드

## 프로젝트 구조

```
pbl_framework/
├── .env                    # API 키 (git에 올리지 않음)
├── config.py               # 설정 (반 추가 시 여기만 수정)
├── pipeline.py             # 5단계 파이프라인 (메인)
├── stability_test.py       # 추출 안정성 테스트 (타당화)
├── requirements.txt        # 패키지 목록
├── data/                   # 데이터 폴더
│   ├── classA.xlsx
│   ├── classB.xlsx
│   ├── classC.xlsx         # 추가 반
│   ├── classD.xlsx
│   └── classE.xlsx
└── outputs/                # 결과 (자동 생성)
    ├── step1_extractions.json   # LLM 추출 원본
    ├── step4_metrics.csv        # 구조적 지표
    ├── final_scores.csv         # 최종 점수
    └── evaluation_report.md     # 평가 보고서
```

## 실행 순서

### 1. 환경 설정

```bash
cd pbl_framework
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일을 열어 API 키를 입력합니다:
```
ANTHROPIC_API_KEY=sk-ant-api03-실제키
```

### 3. 데이터 배치

`data/` 폴더에 classA.xlsx ~ classE.xlsx를 넣습니다.

### 4. 반 추가

`config.py`의 `CLASS_FILES`에 파일을 추가합니다:
```python
CLASS_FILES = {
    "A": DATA_DIR / "classA.xlsx",
    "B": DATA_DIR / "classB.xlsx",
    "C": DATA_DIR / "classC.xlsx",  # 주석 해제
    "D": DATA_DIR / "classD.xlsx",
    "E": DATA_DIR / "classE.xlsx",
}
```

### 5. 파이프라인 실행

```bash
python pipeline.py
```

첫 실행 시 LLM API를 호출하여 추출합니다 (약 1~2분).
결과는 `outputs/step1_extractions.json`에 저장되며,
재실행 시 기존 결과를 자동으로 재사용합니다.

재추출이 필요하면 `outputs/step1_extractions.json`을 삭제하고 재실행합니다.

### 6. 추출 안정성 테스트 (타당화)

```bash
python stability_test.py
```

5명 × 5회 반복 = 25회 API 호출 (약 $0.01)

## 비용 참고

| 작업 | 모델 | 호출 수 | 예상 비용 |
|------|------|---------|----------|
| 전체 추출 (100명) | Haiku, temp=0 | 100회 | ~$0.04 |
| 안정성 테스트 (5명×5회) | Haiku, temp=0 | 25회 | ~$0.01 |
| **합계** | | **125회** | **~$0.05** |

## 출력물 설명

- **step1_extractions.json**: 학생별 엔티티/관계 원본 (LLM 출력 그대로)
- **step4_metrics.csv**: 비율 기반 구조적 지표 + 길이 독립성 검증 결과
- **final_scores.csv**: 반 내 백분위 포함 최종 점수
- **evaluation_report.md**: 반별/개인별 보고서 (Markdown)
