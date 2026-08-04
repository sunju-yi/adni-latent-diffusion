# ==============================================================================
# ADNIMERGE.rda -> 인구통계 공변량 CSV 추출
# ==============================================================================
# 사용법 (R 콘솔 또는 RStudio에서):
#   1) 아래 rda_path를 실제 ADNIMERGE.rda 경로로 수정
#   2) source("extract_demographics.R")  또는 전체 실행
#
# 출력: demographics_baseline.csv
#   RID, AGE, PTGENDER, PTEDUCAT, PTMARRY, APOE4
#
# 설계 의도:
#   - 인구통계는 시간에 따라 거의 변하지 않는 값(baseline 특성)이므로
#     환자당 "최초 방문(baseline, VISCODE=='bl') 1행"만 남깁니다.
#   - AGE는 ADNIMERGE의 baseline age 컬럼을 그대로 사용합니다
#     (방문마다 나이가 달라지는 걸 원하면 AGE + (Years_bl) 로 계산 가능;
#      필요하시면 알려주세요, 스크립트를 확장하겠습니다).
#   - APOE4는 유전 정보이지만 편의상 여기서 함께 추출합니다. Python 쪽에서는
#     demographics 벡터에 포함시켜 "배경변수" 취급을 하되, 필요하면
#     genetics 모달리티 쪽으로 옮겨 쓸 수도 있습니다 (README 참고).

# --- 0. 경로 설정 (반드시 수정) ---
rda_path <- "ADNIMERGE.rda"           # 실제 .rda 파일 경로로 변경
output_csv <- "demographics_baseline.csv"

# --- 1. .rda 로드 ---
# .rda는 저장 당시의 변수명(예: adnimerge, ADNIMERGE 등)을 그대로 복원합니다.
# 변수명이 스크립트마다 다를 수 있어 로드 후 자동으로 data.frame을 찾습니다.
loaded_names <- load(rda_path)
cat("불러온 객체:", paste(loaded_names, collapse = ", "), "\n")

df_candidates <- Filter(function(n) is.data.frame(get(n)), loaded_names)
if (length(df_candidates) == 0) {
  stop("로드된 객체 중 data.frame이 없습니다. rda_path를 확인하세요.")
}
adnimerge <- get(df_candidates[1])
cat("사용할 데이터프레임:", df_candidates[1], " (", nrow(adnimerge), "행)\n")

# --- 2. 필요한 컬럼 확인 ---
required_cols <- c("RID", "VISCODE", "AGE", "PTGENDER", "PTEDUCAT", "PTMARRY", "APOE4")
missing_cols <- setdiff(required_cols, colnames(adnimerge))
if (length(missing_cols) > 0) {
  cat("⚠ 다음 컬럼이 없습니다 (버전에 따라 이름이 다를 수 있음):",
      paste(missing_cols, collapse = ", "), "\n")
  cat("실제 컬럼 목록에서 비슷한 이름을 찾아 수동으로 매핑해주세요:\n")
  print(grep("AGE|GENDER|EDUC|MARRY|APOE", colnames(adnimerge), value = TRUE, ignore.case = TRUE))
}
required_cols <- intersect(required_cols, colnames(adnimerge))

# --- 3. baseline(VISCODE == 'bl') 행만 추출, 환자당 1행 ---
df_bl <- adnimerge[adnimerge$VISCODE == "bl", required_cols]
df_bl <- df_bl[!duplicated(df_bl$RID), ]

cat("Baseline 인구통계 확보 환자 수:", nrow(df_bl), "\n")

# --- 4. 범주형 -> 숫자 인코딩 (Python 쪽 로더가 바로 쓸 수 있도록) ---
if ("PTGENDER" %in% colnames(df_bl)) {
  df_bl$PTGENDER <- ifelse(trimws(as.character(df_bl$PTGENDER)) == "Male", 1, 0)
}
if ("PTMARRY" %in% colnames(df_bl)) {
  # Married=1, 그 외(Widowed/Divorced/Never married/Unknown 등)=0
  df_bl$PTMARRY <- ifelse(trimws(as.character(df_bl$PTMARRY)) == "Married", 1, 0)
}

# --- 5. 저장 ---
write.csv(df_bl, output_csv, row.names = FALSE)
cat("✅ 저장 완료:", output_csv, "\n")
