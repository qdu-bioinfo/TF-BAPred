# TF-BAPred: a universal Bioactive Peptides Predictor Integrating Multiple Feature Representations

Bioactive peptides play crucial roles in various biological processes, showcasing significant therapeutic
potential. Predicting these peptide sequences constitute essential steps in understanding their functions
and harnessing their benefits in medical applications. However, this task is complicated by the diversity
and complexity of bioactive peptides. The key to addressing this challenge lies in extracting effective
features from these peptide sequences. Here, we develop TF-BAPred, a framework for universal peptide
prediction incorporating multiple feature representations. TF-BAPred feeds features into three parallel
modules: a novel feature proposed by this study involves using a fixed-scale vector graph to capture
the global structural patterns of each peptide sequence, an automatic feature recognition module based
on temporal convolutional network to extract temporal features, and a module that integrates multiple
widely-used features, including AAC, DPC, BPF, RSM, and CKSAAGP. We evaluated the performance
of TF-BAPred and other peptide predictors on different types of peptides including anticancer peptides,
antimicrobial peptides, and cell-penetrating peptides. Experimental results demonstrate that TCN-
BAPred displays strong generalization and robustness in predicting various types of peptide sequences,
highlighting its potential for applications in biomedical engineerin



# The dependencies required for TF-BAPred to run

python 3.9.12

tensorflow 2.12.0

keras 2.12.0 

sklearn 1.2.2

numpy 1.22.0

# Instructions for using TF-BAPred
(1)Store the bioactive peptide sequence in a fasta file(There are several demonstration data in the file)

(2)Set the fasta file address in the dataloader section of the code

(3)Adjust parameter settings

(4)Run TF-BAPred

