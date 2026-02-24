package com.rankinggam.inference;

/**
 * Interface for computing set-dependent (groupwise) features.
 *
 * <p>Groupwise features measure the marginal contribution of a candidate item
 * relative to the already-selected set. Examples: category novelty, brand novelty,
 * visual diversity, price spread, attribute coverage.
 *
 * <p>Implementors provide domain-specific feature computation matching the
 * diversity towers in the model. The returned feature array must align with
 * the diversity towers (same order and count).
 *
 * @see SubmodularGamReranker
 */
public interface GroupwiseFeatureComputer {

    /**
     * Compute groupwise features for a single candidate given the selected set.
     *
     * @param candidateIdx    index of the candidate item in the feature matrix
     * @param selectedIndices indices of already-selected items (may be length 0)
     * @param features        [numDocs][numFeatures] full feature matrix
     * @return groupwise feature values, one per diversity tower
     */
    double[] compute(int candidateIdx, int[] selectedIndices, double[][] features);

    /** Number of groupwise features produced (must match diversity tower count). */
    int numFeatures();
}
