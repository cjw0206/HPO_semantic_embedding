import pandas as pd
import ast
import json
import numpy as np


# 경로는 필요에 맞게 수정해서 사용하세요
HPO_EMB_CSV = "data/hpo_omim_embedding.csv"
PHENOTYPE_HPOA = "data/phenotype.hpoa"
OMIM_ID_TXT = "data/omim_ids.txt"
# POOLING = "mean"
POOLING = "no"
OUTPUT_CSV = f"data/omim_semantic_{POOLING}_pool_embeddings.csv"


def load_hpo_embeddings(hpo_emb_csv_path: str) -> dict:
    """
    hpo_omim_embedding.csv 를 읽어서
    HPO ID(HP:xxxx) -> 임베딩 (list[float]) 딕셔너리로 반환.

    hpo_omim_embedding.csv 형식:
        HPO_id,emb
        HP:0000001,"[-0.18, 0.26, ...]"
        ...
    """
    df = pd.read_csv(hpo_emb_csv_path)

    hpo_to_vec = {}
    for _, row in df.iterrows():
        hpo_id = row["HPO_id"] 
        emb_str = row["emb"]
        # "[a, b, c, ...]" 문자열을 list[float]로 파싱
        vec = ast.literal_eval(emb_str)
        hpo_to_vec[hpo_id] = vec

    return hpo_to_vec


def load_phenotype_hpoa(phenotype_path: str) -> pd.DataFrame:
    """
    phenotype.hpoa 파일을 읽어서 DataFrame으로 반환.
    주석(#) 라인은 무시하고, 탭 구분자로 읽는다.
    """
    df = pd.read_csv(
        phenotype_path,
        sep="\t",
        comment="#",      # '#' 으로 시작하는 주석 라인 무시
        dtype=str         # 모든 컬럼을 문자열로 읽기
    )
    # 필요한 컬럼만 남겨두기
    required_cols = ["database_id", "hpo_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"phenotype.hpoa 에 {missing} 컬럼이 없습니다.")

    return df[required_cols].copy()


def load_target_omim_ids(txt_path: str) -> list:
    """
    OMIM ID 리스트 파일을 읽는다.
    각 줄에 하나씩:  예) OMIM:619340
    """
    omim_ids = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            omim_ids.append(line)
    return omim_ids


def build_omim_semantic_embeddings(
    phenotype_df: pd.DataFrame,
    hpo_to_vec: dict,
    omim_ids: list
) -> pd.DataFrame:

    rows = []

    # phenotype_df: columns = ["database_id", "hpo_id"]
    for omim_id in omim_ids:
        df_sub = phenotype_df[phenotype_df["database_id"] == omim_id]

        # annotation된 HPO term들 (중복 제거)
        hpo_ids = df_sub["hpo_id"].dropna().unique().tolist()

        # 해당 OMIM ID에 annotation이 전혀 없으면 스킵 (원하면 빈 것도 포함 가능)
        if len(hpo_ids) == 0:
            # rows.append({"omim_id": omim_id, "hpo_ids": "[]", "emb": "[]"})
            continue

        # HPO term 임베딩 모으기
        emb_list = []
        used_hpo_ids = []
        for hpo_id in hpo_ids:
            vec = hpo_to_vec.get(hpo_id)
            if vec is not None:
                emb_list.append(vec)
                used_hpo_ids.append(hpo_id)

        # 해당 OMIM ID의 HPO term 중에서 임베딩이 하나도 없으면 스킵
        if len(emb_list) == 0:
            continue

        # JSON 문자열로 저장 (나중에 쉽게 파싱 가능)
        hpo_ids_json = json.dumps(used_hpo_ids)
        emb_json = json.dumps(emb_list)

        rows.append(
            {
                "omim_id": omim_id,
                "hpo_ids": hpo_ids_json,
                "emb": emb_json,   # (n_terms, 64) 형태의 리스트-오브-리스트
            }
        )

    return pd.DataFrame(rows)

def pool_embeddings(emb_list: list[np.ndarray], mode: str = "mean") -> np.ndarray:
    """
    여러 개의 (64,) 벡터를 하나로 pooling.
    mode: "mean" 또는 "max"
    """
    stack = np.stack(emb_list, axis=0)  # (n_terms, 64)
    if mode == "mean":
        return stack.mean(axis=0)
    elif mode == "max":
        return stack.max(axis=0)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")


def build_omim_pooled_embeddings(
    phenotype_df: pd.DataFrame,
    hpo_to_vec: dict,
    omim_ids: list,
    pooling: str = "mean",
) -> pd.DataFrame:
    """
    주어진 omim_ids 에 대해,
    phenotype.hpoa 로부터 annotation된 HPO term을 찾고,
    각 HPO term의 임베딩을 pooling 해서 (64,) 벡터 하나로 만든다.

    반환 DataFrame 컬럼:
        omim_id : OMIM:xxxxx
        emb     : "[...64 floats...]" 문자열
    """
    rows = []

    for omim_id in omim_ids:
        df_sub = phenotype_df[phenotype_df["database_id"] == omim_id]

        hpo_ids = df_sub["hpo_id"].dropna().unique().tolist()

        if len(hpo_ids) == 0:
            print(f"{hpo_ids} annotation 없음!!!")
            continue

        emb_list = []
        for hpo_id in hpo_ids:
            vec = hpo_to_vec.get(hpo_id)
            if vec is not None:
                emb_list.append(vec)

        if len(emb_list) == 0:
            print(f"{hpo_id} annotation은 있는데 embedding 잆음!!!")
            continue

        pooled_vec = pool_embeddings(emb_list, mode=pooling)  # (64,)

        emb_str = "[" + ", ".join(f"{v:.8f}" for v in pooled_vec.tolist()) + "]"
        rows.append({"omim_id": omim_id, "emb": emb_str})

    return pd.DataFrame(rows)

def main():
    # 1. HPO term 임베딩 로드 (HP:xxxx -> [64차원 벡터])
    hpo_to_vec = load_hpo_embeddings(HPO_EMB_CSV)

    # 2. phenotype.hpoa 로드 (OMIM:xxxx, HP:xxxx 매핑)
    phenotype_df = load_phenotype_hpoa(PHENOTYPE_HPOA)

    # 3. 대상 OMIM ID 리스트 로드
    omim_ids = load_target_omim_ids(OMIM_ID_TXT)


    if POOLING =="no":
        # 선택) 4-1. OMIM ID마다 semantic embedding 생성 -> 모든 term embedding stack함
        result_df = build_omim_semantic_embeddings(
            phenotype_df=phenotype_df,
            hpo_to_vec=hpo_to_vec,
            omim_ids=omim_ids,
        )
    elif POOLING in ["mean", "max"]:
        # 선택) 4-2. pooling embedding 생성 -> omim id 별로 동일한 차원을 return함
        result_df = build_omim_pooled_embeddings(
            phenotype_df=phenotype_df,
            hpo_to_vec=hpo_to_vec,
            omim_ids=omim_ids,
            pooling=POOLING,
        )

    # 5. CSV로 저장
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(result_df)} OMIM embeddings to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
