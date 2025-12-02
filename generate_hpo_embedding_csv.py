import pickle
import pandas as pd

# 파일 경로 설정
emb_path = "data/emb/hpo-terms-64.emd"   # hpo-terms-64.emd 경로
id_dict_path = "data/hpo_id_dict"    # hpo_id_dict 경로
out_csv_path = "data/hpo_omim_embedding.csv"


def load_embeddings(emb_file):
    """
    node2vec 형식의 임베딩 파일을 읽어서
    index -> vector(list[float]) 딕셔너리로 반환.
    첫 줄은 '노드수 차원수' 헤더라서 스킵.
    """
    vectors = {}
    with open(emb_file, "r") as f:
        header = f.readline().strip()      # 예: "19393 64"
        n_nodes, dim = map(int, header.split())

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            idx = int(parts[0])
            vec = list(map(float, parts[1:]))
            if len(vec) != dim:
                raise ValueError(f"Index {idx} has {len(vec)} dims, expected {dim}")
            vectors[idx] = vec

    return vectors, n_nodes, dim


def main():
    # 1. HPO ID -> index 매핑 불러오기
    with open(id_dict_path, "rb") as f:
        hpo_id_dict = pickle.load(f)   # 예: {"HP:0000001": 0, ...}

    # 2. 임베딩 파일 불러오기 (index -> vector)
    vectors, n_nodes, dim = load_embeddings(emb_path)

    # 3. 매핑 결합해서 CSV용 row 만들기
    rows = []
    missing = []

    for hpo_id, idx in hpo_id_dict.items():
        vec = vectors.get(idx)
        if vec is None:
            missing.append((hpo_id, idx))
            continue

        # 두 번째 컬럼 emb는 "[0.001, 0.2, ...]" 이런 문자열 형태로 저장
        emb_str = "[" + ", ".join(f"{v:.8f}" for v in vec) + "]"
        rows.append({"HPO_id": hpo_id, "emb": emb_str})

    if missing:
        print(f"Warning: {len(missing)} IDs had no embedding:", missing[:10])

    # 4. DataFrame으로 만들어서 CSV로 저장
    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    print(f"Saved {len(df)} rows to {out_csv_path}")


if __name__ == "__main__":
    main()
