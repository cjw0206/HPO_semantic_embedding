import pandas as pd
from collections import defaultdict
import pickle
from pathlib import Path


def hpo_csv_trim(csv_path='hp.obo.csv'):
    """
    HPO OBO CSV 파일에서 임베딩/그래프 생성에 필요한 정보만 정리하는 함수.

    - obsolete(폐기된) 항목 제거 (is_obsolete 컬럼이 있을 경우에만)
    - id, is_a 컬럼만 사용
    - 문자열로 저장된 리스트 형태의 is_a를 실제 리스트로 변환
    - 각 HPO term에 대해 0부터 시작하는 index_mapping 부여

    Args:
        csv_path (str): hp.obo에서 파싱한 CSV 파일 경로

    Returns:
        pd.DataFrame: 정리된 HPO term 정보
                      (id, is_a, index_mapping)
    """
    hpo_terms = pd.read_csv(csv_path)

    # is_obsolete 컬럼이 있으면 obsolete 아닌 term만 사용
    if 'is_obsolete' in hpo_terms.columns:
        valid_terms = hpo_terms.loc[hpo_terms['is_obsolete'].isna()].copy()
    else:
        valid_terms = hpo_terms.copy()

    # 필요한 컬럼이 없으면 에러를 바로 터뜨려서 확인
    required_cols = ['id', 'is_a']
    for col in required_cols:
        if col not in valid_terms.columns:
            raise KeyError(f"Column '{col}' not found in HPO CSV: {csv_path}")

    terms_for_node2vec = valid_terms[required_cols].copy()

    # id 정리 (예: "['HP:0000001']" -> "HP:0000001")
    terms_for_node2vec['id'] = terms_for_node2vec['id'].apply(
        lambda x: x.strip("['']") if isinstance(x, str) else x
    )

    # is_a: 문자열로 된 리스트를 실제 리스트로 변환
    # 예: "['HP:0000001','HP:0000118']" -> ["HP:0000001", "HP:0000118"]
    terms_for_node2vec['is_a'] = terms_for_node2vec['is_a'].apply(
        lambda x: x.strip("[']").replace(' ', '').split("','") if isinstance(x, str) else x
    )

    # 인덱스 리셋 후 정수 ID 부여
    terms_for_node2vec.reset_index(inplace=True, drop=True)
    terms_for_node2vec['index_mapping'] = terms_for_node2vec.index

    return terms_for_node2vec


def create_edge_list(terms_for_node2vec):
    """
    HPO term들에 대해 'is_a' 관계만 사용하여 그래프 edge list를 생성.

    노드는 index_mapping (0, 1, 2, ...) 정수 ID를 사용하고,
    방향은 자식 -> 부모 쪽으로 설정.

    Args:
        terms_for_node2vec (pd.DataFrame): hpo_csv_trim 결과 DataFrame

    Returns:
        dict[int, list[int]]: node_index -> [parent_node_index, ...]
    """
    is_a_dict = dict(
        zip(
            terms_for_node2vec['index_mapping'].values,
            terms_for_node2vec['is_a'].values
        )
    )
    id_to_index = dict(
        zip(
            terms_for_node2vec['id'].values,
            terms_for_node2vec['index_mapping'].values
        )
    )

    graph_edges = defaultdict(list)

    # is_a 관계 추가 (자식 -> 부모)
    for idx, is_a_list in is_a_dict.items():
        if isinstance(is_a_list, list):
            for parent_id in is_a_list:
                if isinstance(parent_id, str) and parent_id in id_to_index:
                    graph_edges[idx].append(id_to_index[parent_id])

    return graph_edges


def save_id_mapping(terms_for_node2vec, save_path='hpo_id_dict'):
    """
    HPO term ID -> 정수 index 매핑을 pickle 파일로 저장.

    dict 형식: { "HP:0000001": 0, "HP:0000002": 1, ... }
    """
    id_to_index = dict(
        zip(
            terms_for_node2vec['id'].values,
            terms_for_node2vec['index_mapping'].values
        )
    )
    with open(save_path, 'wb') as fp:
        pickle.dump(id_to_index, fp, protocol=pickle.HIGHEST_PROTOCOL)


def write_edge_list(graph_edges, save_path='graph/hpo-terms.edgelist'):
    """
    HPO 그래프 edge list를 텍스트 파일로 저장.

    한 줄에 "source_node_index  target_node_index" 형식으로 저장.
    (node2vec, DeepWalk 등에서 바로 읽어서 사용할 수 있는 포맷)
    """
    with open(save_path, 'w') as f:
        for node, neighbors in graph_edges.items():
            for nbr in neighbors:
                f.write(f"{node}  {nbr}\n")


if __name__ == '__main__':
    # hp.obo를 파싱해서 만든 CSV 파일 경로 (obo_file_parsing.py 결과)
    folder_path = 'data/'
    csv_path = folder_path + 'hp.obo.csv'

    # 1. CSV 정리
    terms_for_node2vec = hpo_csv_trim(csv_path=csv_path)

    # 2. HP:... -> index 매핑 저장
    save_id_mapping(terms_for_node2vec, save_path=folder_path + 'hpo_id_dict')

    # 3. 그래프 edge 생성 및 저장
    graph_edges = create_edge_list(terms_for_node2vec)
    Path(folder_path + 'graph').mkdir(parents=True, exist_ok=True)
    write_edge_list(graph_edges, save_path=folder_path + 'graph/hpo-terms.edgelist')
