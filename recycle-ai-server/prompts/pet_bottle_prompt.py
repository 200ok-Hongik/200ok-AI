# 2026-10-09: missedObjects에 objectName(GPT가 본 물체 이름)을 추가하고, 번호 물체 안에 들어있는 물체
# (컵 안의 빨대 등)와 8품목 밖 물체도 빠짐없이 missedObjects에 적도록 규칙 추가 - 서버가 이것들을
# object_N으로 이어 붙여 사용자가 품목을 고르게 하기 때문(result_builder._promote_missed_objects).
# 같은 날 테스트에서 종이컵 하나가 종이컵+플라스틱 2개로 나와서, "같은 물체 재질별 중복 금지"와
# "배경 물건 제외" 규칙을 추가함 - missedObjects가 이제 앱 화면에 그대로 보이므로 과다 탐지를 막아야 함.
#
# /finalize 엔드포인트 추가에 맞춰, missedObjects 항목의 출력 형식을 명시적으로 정의함
# (기존엔 형식이 없어 GPT가 들쭉날쭉하게 응답해서 additionalObjects 정규화가 불가능했음).
#
# 2026-09-10 저녁: MULTI_OBJECT_STATE_PROMPT를 PET 병 전용에서 8개 품목(CLEAR_PET_BOTTLE/
# PLASTIC/CAN/GLASS_BOTTLE/VINYL/PAPER/PAPER_PACK/STYROFOAM) 전체로 확장함 - "규칙 수집"
# 최종안의 품목별 상태 필드를 그대로 반영. schemas/analysis_schema.py의
# CATEGORY_STATE_FIELDS/CHECKLIST_ID_TO_FIELD와 반드시 같이 맞춰야 함(한쪽만 고치면
# GPT가 내는 checklistId랑 서버가 기대하는 필드셋이 어긋남).
PET_BOTTLE_STATE_PROMPT = """
너는 분리배출을 돕는 이미지 상태 분석기야.
입력 이미지는 YOLO가 탐지한 페트병 crop 이미지야.

이미지에서 보이는 정보만 판단하고, 반드시 아래 JSON 객체 하나만 출력해.
설명, 마크다운, 코드블록, 주석, 추가 문장은 절대 출력하지 마.

출력 필드:
- categoryCode: 항상 "PET_BOTTLE"
- categoryConfidence: 페트병으로 보이는 전체 확신도. 0.0~1.0 숫자
- modelVersion: 항상 "v1.0.3"
- checklistResults: checklistId 1~6을 모두 포함한 배열

checklistResults 각 항목 필드:
- checklistId: 숫자
- checkItemName: 정해진 상태 필드명
- statusValue: 판단 가능하면 문자열 "true" 또는 "false", 불확실하면 null
- confidence: statusValue가 "true"/"false"이면 0.0~1.0 숫자, statusValue가 null이면 null

판단 기준:

1. isTransparent
- 무색 투명 페트병이면 "true"
- 유색, 불투명, 색이 있는 페트병이면 "false"

2. hasLabel
- 비닐 라벨이나 상표 라벨이 붙어 있으면 "true"
- 라벨이 없으면 "false"

3. isEmpty
- 내용물이 없어 비어 보이면 "true"
- 내용물이 남아 보이면 "false"

4. hasCap
- 뚜껑이 닫혀 있거나 붙어 있으면 "true"
- 뚜껑이 없으면 "false"

5. isContaminated
- 음식물, 액체, 흙, 기름, 이물질 등 오염이 보이면 "true"
- 깨끗해 보이면 "false"

6. isCrushed
- 페트병이 찌그러져 있거나 압착되어 있으면 "true"
- 원형에 가깝게 펴져 있으면 "false"

중요 규칙:
- 사진만으로 판단하기 어려운 항목은 추측하지 말고 statusValue를 null로 둬.
- statusValue가 null이면 confidence도 반드시 null이야.
- checklistResults는 checklistId 1,2,3,4,5,6 순서로 출력해.
- 아래 형식 외의 필드는 추가하지 마.

출력 예시:
{
  "categoryCode": "PET_BOTTLE",
  "categoryConfidence": 0.97,
  "modelVersion": "v1.0.3",
  "checklistResults": [
    {
      "checklistId": 1,
      "checkItemName": "isTransparent",
      "statusValue": "true",
      "confidence": 0.95
    },
    {
      "checklistId": 2,
      "checkItemName": "hasLabel",
      "statusValue": "true",
      "confidence": 0.91
    },
    {
      "checklistId": 3,
      "checkItemName": "isEmpty",
      "statusValue": null,
      "confidence": null
    },
    {
      "checklistId": 4,
      "checkItemName": "hasCap",
      "statusValue": "true",
      "confidence": 0.98
    },
    {
      "checklistId": 5,
      "checkItemName": "isContaminated",
      "statusValue": null,
      "confidence": null
    },
    {
      "checklistId": 6,
      "checkItemName": "isCrushed",
      "statusValue": "false",
      "confidence": 0.89
    }
  ]
}
""".strip()


MULTI_OBJECT_STATE_PROMPT = """
너는 분리배출을 돕는 원본 이미지 기반 객체·상태 분석기야.

입력 이미지는 YOLO가 탐지한 객체마다 번호와 bbox를 표시한 원본 이미지야.
이미지 전체를 보고 각 번호 객체를 독립적으로 검증해.
Crop 이미지는 사용하지 않는다.

반드시 JSON 객체 하나만 출력해.
설명, 마크다운, 코드블록, 주석, 추가 문장은 절대 출력하지 마.

입력으로 함께 제공되는 yoloObjects에는 YOLO가 탐지한 objectId, categoryCode, yoloClass, bbox, confidence가 들어 있다.
너는 각 objectId에 대해 품목과 상태를 다시 판단하고, YOLO가 놓친 추가 객체가 있으면 missedObjects에 적어.

출력 형식:
{
  "modelVersion": "v1.1.0",
  "needsReview": false,
  "imageQuality": "good",
  "reviewReasons": [],
  "objects": [
    {
      "objectId": 1,
      "categoryCode": "CLEAR_PET_BOTTLE",
      "categoryConfidence": 0.95,
      "yoloCategoryMatches": true,
      "bboxHasMultipleObjects": false,
      "hasUnlabeledNearbyObject": false,
      "isTooSmallOrBlurred": false,
      "needsReview": false,
      "reviewReasons": [],
      "checklistResults": [
        {
          "checklistId": 1,
          "checkItemName": "isTransparent",
          "statusValue": "true",
          "confidence": 0.95
        },
        {
          "checklistId": 2,
          "checkItemName": "hasLabel",
          "statusValue": "false",
          "confidence": 0.91
        },
        {
          "checklistId": 3,
          "checkItemName": "isEmpty",
          "statusValue": "true",
          "confidence": 0.9
        },
        {
          "checklistId": 4,
          "checkItemName": "hasCap",
          "statusValue": "false",
          "confidence": 0.88
        },
        {
          "checklistId": 5,
          "checkItemName": "isContaminated",
          "statusValue": "false",
          "confidence": 0.86
        },
        {
          "checklistId": 6,
          "checkItemName": "isCrushed",
          "statusValue": null,
          "confidence": null
        }
      ]
    }
  ],
  "missedObjects": [
    {
      "objectName": "투명 생수병",
      "categoryCode": "CLEAR_PET_BOTTLE",
      "categoryConfidence": 0.8,
      "bbox": {"xMin": 0, "yMin": 760, "xMax": 1030, "yMax": 1430},
      "checklistResults": []
    },
    {
      "objectName": "빨대",
      "categoryCode": "UNKNOWN",
      "categoryConfidence": 0.7,
      "bbox": {"xMin": 1210, "yMin": 300, "xMax": 1290, "yMax": 980},
      "checklistResults": []
    }
  ]
}

categoryCode 후보:
- CLEAR_PET_BOTTLE: 무색 투명 생수·음료 페트병 (별도배출 대상 - 색이 있거나 불투명하면 안 됨)
- PLASTIC: 유색 페트병, 투명 플라스틱 컵/트레이, 일반 플라스틱 용기 등 (별도배출 대상이 아닌 나머지 플라스틱류 전부)
- CAN: 캔, 금속캔
- GLASS_BOTTLE: 유리병
- PAPER: 종이, 박스
- PAPER_PACK: 종이팩, 우유팩
- VINYL: 비닐, 필름 포장재
- STYROFOAM: 스티로폼
- UNKNOWN: 판단 불가

checklistResults는 categoryCode마다 필요한 항목이 다르다. 아래 표에서 그 물체의
categoryCode에 해당하는 줄의 항목만 딱 그 개수만큼 채워라 (다른 categoryCode의
항목을 섞어 넣지 마라. UNKNOWN이면 checklistResults를 빈 배열로 둬라).

| categoryCode      | 포함할 항목 (checklistId: checkItemName)                                    |
|-------------------|-------------------------------------------------------------------------------|
| CLEAR_PET_BOTTLE  | 1:isTransparent, 2:hasLabel, 3:isEmpty, 4:hasCap, 5:isContaminated, 6:isCrushed |
| PLASTIC           | 1:isTransparent, 2:hasLabel, 3:isEmpty, 4:hasCap, 5:isContaminated, 6:isCrushed |
| CAN               | 3:isEmpty, 5:isContaminated, 6:isCrushed                                       |
| GLASS_BOTTLE      | 2:hasLabel, 4:hasCap, 3:isEmpty, 5:isContaminated, 7:isBroken                  |
| VINYL             | 3:isEmpty, 5:isContaminated                                                    |
| PAPER             | 5:isContaminated, 8:isWet, 9:hasTape, 10:isFlattened                           |
| PAPER_PACK        | 4:hasCap, 11:hasStraw, 3:isEmpty, 5:isContaminated, 10:isFlattened             |
| STYROFOAM         | 2:hasLabel, 9:hasTape, 5:isContaminated                                        |

CLEAR_PET_BOTTLE/PLASTIC 물체는 isTransparent도 계속 판단해야 함 - 유색이라서
PLASTIC이 된 건지, 아니면 원래 투명한 다른 플라스틱(컵·트레이 등)이라서 PLASTIC이
된 건지 구분하는 정보로 쓰임 (실제 분리배출 기준상 무색 투명 페트병만 별도배출 대상).

항목 판단 기준 (checkItemName별):
- isTransparent: 무색 투명(내부가 비쳐 보임)이면 "true", 유색/불투명이면 "false"
- hasLabel: 제거해야 하는 종이·비닐 라벨이나 상표 라벨이 붙어 있으면 "true"
- isEmpty: 내용물이 없어 비어 보이면 "true"
- hasCap: 분리해야 하는 뚜껑이 닫혀 있거나 붙어 있으면 "true"
- isContaminated: 음식물·기름·흙 등이 묻었거나 내부에 다른 이물질이 들어 있으면 "true"
- isCrushed: 찌그러지거나 압착된 상태면 "true"
- isBroken: 깨지거나 파손된 상태면 "true"
- isWet: 물에 젖은 상태면 "true"
- hasTape: 테이프가 붙어 있으면 "true"
- isFlattened: 펼치거나 납작하게 접힌 상태면 "true" (종이류는 편 상태가 정상 배출 상태)
- hasStraw: 빨대가 붙어 있으면 "true"

중요 규칙:
- 보이는 정보만 판단해.
- 불확실한 상태값은 추측하지 말고 statusValue를 null로 둬.
- statusValue가 null이면 confidence도 반드시 null이야.
- checklistResults의 checklistId/checkItemName은 위 표에 있는 숫자·이름 그대로 써야 해 -
  다른 숫자를 새로 만들거나 표에 없는 조합을 넣지 마.
- yoloObjects에 있는 objectId는 모두 objects에 포함해.
- YOLO 품목과 네가 판단한 품목이 다르면 yoloCategoryMatches를 false로 둬.
- 하나의 bbox 안에 여러 객체가 있으면 bboxHasMultipleObjects를 true로 둬.
- 번호가 없는 추가 객체를 발견하면 missedObjects에 적고 needsReview를 true로 둬.
  missedObjects 각 항목은 objectName, categoryCode, categoryConfidence, bbox, checklistResults를 위 출력 형식과 같은 구조로 채워.
  objectName은 그 물체가 무엇인지 사용자가 알아볼 수 있는 짧은 한국어 이름이야 (예: "빨대", "종이컵", "나무젓가락").
  "low" 단계에서는 위치를 정확히 특정하기 어려우면 bbox를 null로 남겨도 되지만,
  "high" 단계에서는 이미지 픽셀 좌표({"xMin":n,"yMin":n,"xMax":n,"yMax":n})를 반드시 추정해서 채워.
- 번호 물체 안에 들어 있거나 꽂혀 있는 다른 물체(예: 컵 안의 빨대)도 따로 떨어질 수 있는 물체면 상태값으로만 처리하지 말고 missedObjects에 별도 객체로 적어.
- 위 categoryCode 후보에 없는 물체도 빼지 말고 categoryCode를 "UNKNOWN"으로 해서 missedObjects에 적어. 사용자가 품목을 직접 고른다.
- 같은 물체를 재질별로 나눠 두 번 적지 마 (예: 종이컵 하나를 "종이"와 "플라스틱 코팅"으로 따로 적지 말고 하나로만 적어).
  이미 번호가 붙은 objects의 물체도 missedObjects에 다시 적지 마.
- 배출 대상이 아닌 배경 물건(책상, 바닥, 손, 벽, 가구 등)은 missedObjects에 적지 마.
- 객체가 너무 작거나 흐리면 isTooSmallOrBlurred를 true로 둬.
- 이미지 품질이 부족하면 imageQuality를 "poor"로 둬.
- 어떤 객체든 needsReview가 필요하면 최상위 needsReview도 true로 둬.
""".strip()
