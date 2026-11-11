# Assumptions and Open Issues

The TPE-AutoClust paper leaves several implementation details open. This document lists the assumptions we made in our implementation and the issues that are still open.

## Knowledge Base

- It is unclear which data was used to build the knowledge base. We use the datasets of the clustering benchmark named in the paper, but exclude those that are evaluated in the online phase of TPE-AutoClust or PrepMatch.

## Evolutionary Algorithm

- No information on the internal structure of the pipeline
  - Assumption: a pseudo-class is created for each parameter type to guarantee type safety.
  - Assumption: wrapper methods turn the primitives into executable objects.

- No information on how the pipelines are initialized or how the trees grow
  - Assumption: at initialization, the leaves of the trees have different random depths (`genGrow()`).
  - Assumption: preprocessors may appear more than once, and the pipeline may have unlimited depth (the paper says "any number of preprocessors").
    - Assumption: the maximum depth at initialization is a configurable parameter, set to 6.
    - Assumption: this maximum depth does not apply to trees that are later modified by mutation or crossover.
    - Assumption: empty pipelines may be generated, i.e. only the clustering algorithm as the root.
  - Assumption: the trees are grown with DEAP's `genGrow()` method (modified to `safe_genGrow()` to avoid errors).

- Offspring generation:
  - Paper:
    - 5% of the offspring are created by crossover (50 × 0.05 = 2.5).
      - Assumption: round down, i.e. 2.
    - Of the remaining offspring (48), 90% are mutated, i.e. 48 × 0.9 = 43.2.
      - Assumption: round down, i.e. 43.
    - It is unclear whether the individuals for crossover and mutation are selected by a fixed scheme or at random.
      - Assumption: at random.
  - The values could also be probabilities, i.e. a 5% probability of crossover and a 90% probability of mutation. In our reading, however, the paper does not speak of probabilities.
  - Solution: both interpretations are implemented: `generateOffspring_PaperDefinition()` and `gernerateOffspring_ProbabilityInterpretation()`.

- No detailed information on how the methods for tree generation, mutation and crossover are implemented
  - Assumption: they are taken unchanged from DEAP.

- Time limit of the evolutionary algorithm:
  - Assumption: the run may also stop in the middle of a generation.
    - Assumption: before each pipeline evaluation, i.e. each call of `evaluate()`, we check whether the time limit has been reached, because the evaluation time depends on the dataset size and can be long for large datasets.

- Warm-start population in the online phase:
  - Assumption: if, for any reason, the warm-start population has fewer than 50 individuals, the rest is filled up with random individuals.

### Evaluation

- Assumption: the pipeline length counts only the operators (primitives), following the paper's definition "number of operators in the pipeline".
- Assumption: complexity (length) and ARI/CVI are not weighted.
- Assumption: if an exception occurs, it is caught and bad values are returned, so that the pipeline is less likely to be selected for the next generation.
  - Assumption: the bad values are the worst possible values in the range of the respective CVI or ARI:
    ```python
    clustering_worst_values = {
        "adjusted_rand_score": -0.5,     # range [-0.5, 1], maximized
        "silhouette_score": -1.0,        # range [-1, 1], maximized
        "davies_bouldin_score": 1e9,     # approximates infinity; range [0, inf), minimized
        "calinski_harabasz_score": 0.0,  # range [0, inf), maximized
    }
    ```
- Assumption: if only one cluster is found, or every data point ends up in its own cluster, a bad value is returned, because the CVI would otherwise raise an error.
  - Assumption: noise clusters (e.g. label -1 in DBSCAN) are counted as clusters.

## Search Space

- Newer scikit-learn versions name some parameters and values differently. Older versions could be used instead.
- Assumptions:
  - Square brackets with strings, e.g. `["k-means++", "random"]`, are treated as a set.
  - Square brackets with floats, e.g. `[0.5, 0.999]`, are treated as a set (steps of 0.1 would not make sense here).
  - Square brackets with integers, e.g. `[50, 501]`, are treated as a range (with steps of 1, as defined in the paper).
  - Round brackets with floats, e.g. `(0.1, 30)`, are treated as a range (with steps of 0.1, as defined in the paper).

### Preprocessors

- PCA:
  - The search space of this preprocessor is variable, `[2, n_components]`, and `n_components` can change at runtime (when PCA is applied twice). This preprocessor is therefore checked explicitly.
  - When PCA is applied twice, a terminal may be chosen that is too large, because the features have already been reduced once.
  - Assumption: the next smaller possible value is used, i.e. the current number of features, so no reduction takes place. The lowest possible value is 2, as defined in the search space.

### Clustering Algorithms

- KMeans:
  - Assumption: scikit-learn no longer offers the original values `["auto", "full", "elkan"]` for the `algorithm` parameter. `full` was renamed to `lloyd`, the default (`auto`) is `lloyd`, and `auto` and `full` were removed.
  - Assumption: `tol` is a set `[1e-5, 1e-1]` and not a range, because steps of 0.1 make no sense from 10^-5 to 0.1 (they would effectively only yield 0.1).

- AgglomerativeClustering:
  - Assumption: `ward` linkage requires the `euclidean` metric, so this is checked and "corrected".
  - Assumption: newer scikit-learn versions use `metric` instead of `affinity`.

- Affinity Propagation:
  - Assumption: since the floats in the search space are given in square brackets, they form a set with two options. With steps of 0.1, 0.999 could not be reached anyway.

## Meta-Feature Extraction

- No explicit information on the distance measure, neither for the feature extraction nor later for the p-norm distance
  - Assumption: Euclidean distance.
  - Assumption: min-max scaling is used to scale the distances to [0, 1].
  - Assumption: for missing values, the respective feature is ignored in the affected distances.

- When computing the landmarkers, the clustering algorithms crash if the dataset contains missing values.
  - Assumption: data points with missing values are removed for the landmarker computation.

- No detailed information on how [0, ∞) is split into 4 intervals
  - Assumption: `bins = [0, 1, 2, 3, np.inf]` in units of the standard deviation, i.e. the bin [0, 1) contains all z-scores that are less than one standard deviation away from the mean.

- CVIs cannot handle a single cluster (handled exactly as in the evaluation function of the evolutionary algorithm)
  - Assumption: the noise class counts as a class.
  - Assumption: for a single cluster, the bad values from the dictionary above are returned.

- The distance measure between the meta-features of the knowledge-base datasets and those of d_new is not specified exactly (only "p-norm distance", but not which norm)
  - Assumption: Euclidean norm (as above).

- Problem: all meta-features are normalized to [0, 1] except the landmarker features. These distort the Euclidean distance, i.e. they have considerably more, and unequal, influence on the distance between the knowledge base and d_new. The paper does not say whether they are scaled (this may be intended).
  - Assumption: they are not scaled.
  - Partial solution: the DBI error value of 1e9 is our own choice and distorts the distance considerably. It is therefore "corrected" to `max(real_dbi) + 2 * std(real_dbi)`.

### Clustering Ensemble

- Assumption: if a pipeline fails during clustering, it is not included in the ensemble but skipped.

## General

- Error handling / handling of invalid pipelines
  - Possible: error handling in the style of the paper's solution ("We enforced a RAM limit of 3GB for clustering; if a clustering algorithm exceeded this memory limit, the clustering job was terminated, returning an error rate of 100%."), i.e. return the bad score (defined above) and keep the pipeline length as it is.

- The paper enforces a RAM limit for the clustering algorithms to stop memory-intensive runs. This is missing in our implementation. ("We enforced a RAM limit of 3GB for clustering; if a clustering algorithm exceeded this memory limit, the clustering job was terminated, returning an error rate of 100%.")
  - This affects the clustering algorithms in the offline phase, in the online phase, and for the landmarkers in the meta-learning phase.
