# Generating Understanding and Interpretation of Multi-Omics Data with an Automated and Generalizable Pipeline

# Objective

We aim to develop a prototype pipeline that rapidly predicts mechanisms driving disease pathogenesis using multi-omics data, demonstrating feasibility with generative AI for uncovering biological mechanisms based on key features from data harmonization.

# Team

## PNNL

- **Samantha Erwin** - AI Research
- **Lisa Bramer** - Domain Expert
- **Daniel Claborne** - AI Engineer
- **Javier Flores** - AI Research
- **Matt Jensen** - AI Engineer

## Domain & Relevant Sectors - Predictive Phenomics and 'Omics Discovery

- **Karl Mueller** - Basic Energy Sciences: Chem, Geo, and Biosciences
- **David Wunschel** - National Security: Chem/Bio

## Reviewer

- Lauren Charles

# Background

With advances in instrumentation, we now collect vast multi-omics datasets that provide insights into biological systems and disease mechanisms.
Current methods are often manual and time-consuming.
Our team has developed a scalable deep learning model for multi-omics data harmonization.
We aim to automate interpretation of predictive features using generative AI, thereby speeding up the discovery of biological mechanisms from existing datasets.

# Approach

We will use our harmonization model to identify key features from multi-omics datasets. Subsequently, we will apply Llama 3, fine-tuned with Retrieval Augmented Fine-Tuning (RAFT) on 'omics-related literature, to interpret these features and elucidate mechanisms driving disease pathogenesis.

# Data Sources

- **Multi-Omics Data**: From PNNL's study on host responses to lethal human viruses.
    - [Publication](https://www.nature.com/articles/s41597-024-03124-3)
    - [PNNL DataHub for Multi-onmics Publication](https://data.pnnl.gov/group/nodes/publication/33863)
- **Literature**: ‘Omics-related publications from PubMed, Medline, and Wikipathways.

# Success Metrics

- **30-day Goal**: Fine-tune Llama 3 with RAFT using relevant literature.
- **60-day Goal**: Demonstrate the model's ability to identify known biological mechanisms.

# Contact

- **Samantha Erwin**: [samantha.erwin@pnnl.gov](mailto:samantha.erwin@pnnl.gov)

# References

1. Erwin, S et al. 2024; doi:10.1101/2023.09.06.556597
2. Lee, C & van der Schaar, M. 2021; doi:10.48550/2102.03014
3. Slenter DN et al. 2018; doi:10.1093/nar/gkx1064
4. Zhang, T et al. 2024; doi:10.48550/2403.10131

# Project High Level

## Model Pipeline

**Step 1 (previous experiment)**

- In: Pathogen name/id
- Model: DeepIMV
- Out: Key macromolecules for predicting infection

**Step 2 (genraitor)**

- In: Macromolecules
- Model: Llama3 RAFT
- Out: Relevant metobolic pathways

**Overall**

- In: Pathogen
- Model: Genraitor
- Out: Metobolic pathways
