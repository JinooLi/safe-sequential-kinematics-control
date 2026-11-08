# Panda Test 1,000장면의 Direct / All-link Sequential 평가

UBAI 작업 1058692의 총 2,000회 평가 및 검증이 완료되었다. 기존 Test 장면, 보관된 제어기와 QP, 120초 제한을 사용했다. 수치 오류도 실패에 포함했다.

| 방법 | 성공 수 / 장면 수 | 성공률 |
|---|---:|---:|
| Direct | 455 / 1,000 | 45.50% |
| All-link Sequential | 552 / 1,000 | 55.20% |

석사논문의 Figure 4.3, Table 4.4, 관련 본문, 데이터 출처와 PDF를 갱신했다. 이 Test 분할은 개발 중 이미 확인한 데이터이므로 새 독립 평가셋으로 해석하지 않는다.

- `comparison.csv`: 장면별 두 제어기의 결과
- `analysis.json`: 성공률과 종료 사유 집계
- `test_comparisons.json`: 기존 두 링크 결과와의 대응 비교 및 신뢰구간
- `completion_verification.json`: 검증 기록과 UBAI 원본 위치
- `thesis_update_complete.json`: 원고·그림·PDF 변경 파일의 해시와 백업 위치
