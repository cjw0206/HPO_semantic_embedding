import pandas as pd

# 입력 / 출력 파일 경로 설정
in_csv = "../datasets/ADSLab_dataset/hpo-terms/omim_semantic_mean_pool_embeddings.csv"      # omim_id,emb 형식 CSV
out_csv = "../datasets/ADSLab_dataset/hpo-terms/disease_semantic_mean_pool_embeddings64.csv"         # 최종 출력 CSV

# 1) OMIM 임베딩 파일 읽기
df = pd.read_csv(in_csv)   # columns: omim_id, emb (가정)

if not {"omim_id", "emb"}.issubset(df.columns):
    raise ValueError(f"{in_csv} 에 'omim_id', 'emb' 컬럼이 필요합니다. 현재 컬럼: {df.columns.tolist()}")

# 2) 너가 준 병 이름 리스트 (이 순서대로 위에서부터 사용)
disease_names = [
    "hypertension",
    "diabetes",
    "cancer",
    "lung",
    "heart",
    "stroke",
    "mental",
    "arthr",
    "memory",
]

# 3) 길이 맞추기: OMIM 임베딩 개수와 병 이름 개수 중 더 작은 쪽까지만 사용
n = min(len(df), len(disease_names))
df = df.iloc[:n].reset_index(drop=True)

# 4) 새로운 DataFrame 구성
#    id      : 네가 준 병 이름
#    feature : 기존 emb 값 그대로
df_out = pd.DataFrame({
    "id": disease_names[:n],
    "feature": df["emb"],
})

# 5) CSV로 저장
df_out.to_csv(out_csv, index=False, encoding="utf-8")
print(f"Saved {len(df_out)} rows to {out_csv}")
