# Osei
This repository contains code to generate the results from the manuscript, "Deep learning chromatin profiles reveals the cis-regulatory sequence code of the rice genome."
## Overview
Recent advances in deep learning have led to the creation of models that generate summarized sequence representations of genomic regulatory activity, offering a functional perspective on regulatory DNA variation in the human genome. Building on this, we extend the approach to the rice genome to explore its cis-regulatory sequence code and assess its transferability across crop species. Here, we developed a deep learning sequence model (`Osei`), which is based on the Sei framework, to predict diverse chromatin profiles in rice.

This code has been tested on Python 3.6, and includes a number of Python scripts and Python/R Jupyter notebooks. Please set up a conda environment, install the packages listed in the `requirements.txt` file, and also install the R kernel for Jupyter notebook. Example commands:
```bash
conda create --name=Osei python=3.6
conda activate Osei
conda install jupyter
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=Osei
conda install -c r r-irkernel
conda install --file requirements.txt -c anaconda -c conda-forge -c bioconda -c pytorch -c intel
```
Some of the python notebooks also call R with rpy2, therefore the R dependencies need to be installed. The R package dependencies are `data.table`, `ggplot2`, `ggrepel`, `patchwork`, `shades`, and `plyr`.
## Database
[Osei](https://biobigdata.nju.edu.cn/Osei/home)

**NOTE**:You can access our database for the data and model used in these analyses.
## Steps in pre-processing the genome
You can run the following program in the shell，Here's an example using rice.
```bash
species=oryza_sativa         # Species name
window_size=1024     # Window size
step_size=128       # Slide step

# Generating intervals...
bedtools makewindows -g ../fasta/${species}.size -w ${window_size} -s ${step_size} > ${species}_${window_size}_${step_size}s_intervals.bed
#Extracting sequences from genome...
bedtools getfasta -fi ../fasta/${species}.fa -bed ${species}_${window_size}_${step_size}s_intervals.bed > ${species}_${window_size}_${step_size}s_intervals.fa
#Converting sequences to single line format...
awk '/^>/{if(seq) print seq; print; seq=""; next} {seq=seq$0} END{if(seq) print seq}' ${species}_${window_size}_${step_size}s_intervals.fa > ${species}_${window_size}_${step_size}s_intervals_single_line.fa
#Filtering sequences...(There may be unassembled sequences)
faFilter -minSize=${window_size} -maxN=0 ${species}_${window_size}_${step_size}s_intervals_single_line.fa stdout | \\
awk '/^>/{if($0 ~ /Un/ || $0 ~ /Sy/) {skip=1} else {skip=0} } !/^>/ && !skip {print} /^>/ && !skip {print}' > ${species}_${window_size}_${step_size}s_intervals_single_line_filtered_temp.fa
#Converting filtered sequences to single line format...
awk '/^>/{if(seq) print seq; print; seq=""; next} {seq=seq$0} END{if(seq) print seq}' ${species}_${window_size}_${step_size}s_intervals_single_line_filtered_temp.fa > ${species}_${window_size}_${step_size}s_intervals_single_line_filtered.fa
rm ${species}_${window_size}_${step_size}s_intervals_single_line_filtered_temp.fa
#Generate bed files for analysis
awk '/^>/{if(seq) print a[1] "\\t" a[2] "\\t" a[3] "\\t" header; header=$0; sub(/^>/, "", header); split(header, a, "[:-]"); seq=""; next} {seq=seq$0} END {if(seq) print a[1] "\\t" a[2] "\\t" a[3] "\\t" header}' ${species}_${window_size}_${step_size}s_intervals_single_line_filtered.fa > ${species}_${window_size}_${step_size}s_intervals_single_line_filtered.bed
```
If you want to collect peak data yourself to train your own species, you need to do overlap on the processed genome bed file and the file of peak signals.
## Code for results
The directories correspond to the following figures/analyses:
-   `crossSpecies`:Predictions are made across species using the model and then the predictions are compared to predefined states.
-   `Experiment`: Correlation structure of model predictions matches the correlation structure of the targets.
-   `enrichmentHeatmap`:Log fold-change enrichment heatmaps.
-   `evolutionaryConstrain`:Regulatory sequence classes are under evolutionary constraints.
-   `performance_curve`:Chromatin profile model performance.
-   `transferModel`:Load a pre-trained model, modify its architecture to retrain the model.
-   `variationEva`:Feature vectors are generated from single nucleotide mutations, predicted using a model, followed by clustering analysis of the variants, and finally visualized to show the impact of the variants in different functional regions.
-   `visualize_UMAP`:The dimensionality is reduced and clustered, and then visualized with UMAP.
