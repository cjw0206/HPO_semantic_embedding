import pandas as pd
import ast
import json
import numpy as np


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
        vec = ast.literal_eval(emb_str)
        hpo_to_vec[hpo_id] = vec

    return hpo_to_vec


def load_phenotype_hpoa(phenotype_path: str) -> pd.DataFrame:
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

        if len(hpo_ids) == 0:
            print(f"{hpo_ids} annotation 없습니다!!!")
            continue

        # HPO term 임베딩 모으기
        emb_list = []
        used_hpo_ids = []
        for hpo_id in hpo_ids:
            vec = hpo_to_vec.get(hpo_id)
            if vec is not None:
                emb_list.append(vec)
                used_hpo_ids.append(hpo_id)

        if len(emb_list) == 0:
            print(f"{hpo_id} annotation은 있는데 embedding 잆습니다!!!")
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

    rows = []

    for omim_id in omim_ids:
        df_sub = phenotype_df[phenotype_df["database_id"] == omim_id]

        hpo_ids = df_sub["hpo_id"].dropna().unique().tolist()

        if len(hpo_ids) == 0:
            print(f"{hpo_ids} annotation 없습니다!!!")
            continue

        emb_list = []
        for hpo_id in hpo_ids:
            vec = hpo_to_vec.get(hpo_id)
            if vec is not None:
                emb_list.append(vec)

        if len(emb_list) == 0:
            print(f"{hpo_id} annotation은 있는데 embedding 잆습니다!!!")
            continue

        pooled_vec = pool_embeddings(emb_list, mode=pooling)  # (64,)

        emb_str = "[" + ", ".join(f"{v:.8f}" for v in pooled_vec.tolist()) + "]"
        rows.append({"omim_id": omim_id, "emb": emb_str})

    return pd.DataFrame(rows)

def main():
    hpo_to_vec = load_hpo_embeddings(HPO_EMB_CSV)

    phenotype_df = load_phenotype_hpoa(PHENOTYPE_HPOA)

    omim_ids = load_target_omim_ids(OMIM_ID_TXT)


    if POOLING =="no":
        # 선택) 1. OMIM ID마다 semantic embedding 생성 -> 모든 term embedding stack함
        result_df = build_omim_semantic_embeddings(
            phenotype_df=phenotype_df,
            hpo_to_vec=hpo_to_vec,
            omim_ids=omim_ids,
        )
    elif POOLING in ["mean", "max"]:
        # 선택) 2. pooling embedding 생성 -> omim id 별로 동일한 차원을 return함
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
