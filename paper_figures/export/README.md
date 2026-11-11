# Figures of the paper

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace paper_figures.ipynb
```

Writes the figures and tables of the evaluation to `output/`:

- `all_algorithms_median_bars4.pdf`: median ARI for four clustering algorithms
- `k_p_comb3.pdf`: median ARI for different k and p
- `baselines_small_5.pdf`: ARI distribution on the synthetic datasets
- `tab_baselines.tex`: ARI comparison of PrepMatch and the baselines
- `tab_ppevalgroups.tex`: ARI change per data characteristic
