## Semantic Embedding Background

This project adapts the semantic embedding approach introduced in the **TransformerGO** paper,  
where proteins are embedded using annotations from the Gene Ontology (GO).

**Paper Info**
   - Ieremie et al., "TransformerGO: predicting protein–protein interactions by modelling the attention between sets of gene ontology terms." Bioinformatics 38.8 (2022): 2269-2277.
   - doi: https://doi.org/10.1093/bioinformatics/btac104

In our case, we apply the same principle to the **Human Phenotype Ontology (HPO)**.  
Because **diseases act as the annotation source in HPO**, we generate **semantic embeddings for diseases** instead of proteins.  
These embeddings capture hierarchical relationships and semantic meaning based on HPO structure and annotation patterns.

The code is based on the implementation provided by **TransformerGO**, and it has been refactored to focus specifically on generating semantic embeddings.

## Dependencies

Install the only required package:

```bash
pip install node2vec
```


## Execution Order

1. **`obo_file_parsing_HPO.py`**  
   Parses the HPO `.obo` file and converts it into a structured dictionary and CSV.  
   **Output:**
   - `hp.obo.csv`

3. **`node2vec_embeddings_HPO.py`**  
   Builds the HPO graph and prepares input data for node2vec.  
   **Outputs:**  
   - `hpo_id_dict`  
   - `hpo-terms.edgelist`

4. **`node2vec_library_hpo.py`**  
   Trains node2vec embeddings using the generated graph.  
   **Output:**
   - `emb/hpo-terms-64.emd`

6. **`generate_hpo_embedding_csv.py`**  
   Converts trained embeddings into a CSV file.  
   **Output:**
   - `hpo_omim_embedding.csv`

8. **`generate_semantic_embedding.py`**  
   Generates semantic embeddings based on the HPO structure.

   **Description**  
   Since semantic embeddings depend on which HPO terms a disease is annotated with,  
   the dimensionality can vary across diseases.  
   To address this, pooling methods are provided for embedding dimension alignment.  
   The `POOLING` variable supports `['mean', 'max', 'no']`,  
   corresponding to **mean pooling**, **max pooling**, and **no pooling**, respectively.

   **Input**  
   - `omim_ids.txt` — contains one OMIM ID per line.  
     If an OMIM ID is missing from HPO annotations or does not exist in the mapping,  
     the script will print a message indicating that the ID is not found.

   **Output**  
   - `omim_semantic_{POOLING}_pool_embeddings.csv`



