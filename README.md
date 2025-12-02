## Semantic Embedding Background

This project adapts the semantic embedding approach introduced in the **TransformerGO** paper,  
where proteins are embedded using annotation signals from the Gene Ontology (GO).
Ieremie et al., "TransformerGO: predicting protein–protein interactions by modelling the attention between sets of gene ontology terms." Bioinformatics 38.8 (2022): 2269-2277.
doi: https://doi.org/10.1093/bioinformatics/btac104

In our case, we apply the same principle to the **Human Phenotype Ontology (HPO)**.  
Because **diseases act as the annotation source in HPO**, we generate **semantic embeddings for diseases** instead of proteins.  
These embeddings capture hierarchical relationships and semantic meaning based on HPO structure and annotation patterns.

The code is based on the implementation provided by **TransformerGO**, and it has been refactored to focus specifically on generating semantic embeddings.


## Execution Order

Run the scripts in the following order:

1. **`obo_file_parsing_HPO.py`**  
   Parses the HPO `.obo` file and converts it into a structured dictionary and CSV.

2. **`node2vec_embeddings_HPO.py`**  
   Builds the HPO graph and prepares input data for node2vec.

3. **`node2vec_library_hpo.py`**  
   Trains node2vec embeddings using the generated graph.

4. **`generate_hpo_embedding_csv.py`**  
   Converts trained embeddings into a CSV file.

5. **`generate_semantic_embedding.py`**  
   Generates semantic embeddings based on the HPO structure.

