import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Literal

def bootstrap_topk_pairs_from_embeddings(
        mentor_embeddings: npt.NDArray[np.float32],
        mentee_embeddings: npt.NDArray[np.float32],
        k: int = 3,
        method: Literal['topk', 'hungarian_plus_topk'] = 'hungarian_plus_topk'
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        
        """
        Bootstrap top-K positive pairs for each mentor using raw embeddings.
        
        This provides multiple positive candidates per mentor, reducing bootstrap noise
        and improving gradient diversity during training.
        
        Args:
            mentor_embeddings: (N_mentors, D) raw combined embeddings
            mentee_embeddings: (N_mentees, D) raw combined embeddings
            k: Number of positive candidates per mentor (default: 3)
            method: 
                - 'topk': Simply take top-K most similar mentees
                - 'hungarian_plus_topk': Use Hungarian for best match, then add k-1 neighbors
        
        Returns:
            pos_pairs: (N_mentors, k) array where pos_pairs[i, :] are the top-k mentees for mentor i
            unused_mentee_indices: (N_unused,) array of mentee indices not in any top-k
        
        Example:
            >>> pos_pairs, unused = bootstrap_topk_pairs_from_embeddings(
            ...     mentor_emb, mentee_emb, k=3
            ... )
            >>> pos_pairs.shape
            (500, 3) # 500 mentors each has k=3 top mentees
            >>> pos_pairs[0]  # Top 3 mentees for 0th mentor 
            array([342, 891, 567])
        """
        n_mentors = len(mentor_embeddings)
        n_mentees = len(mentee_embeddings)

        # Validation
        if n_mentees < k:
            raise ValueError(f"Need at least {k} mentees, got {n_mentees}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        # Compute cosine similarity (embeddings should be L2-normalized from S-BERT)
        similarity_matrix = np.dot(mentor_embeddings, mentee_embeddings.T)

        if method == 'topk':
            # Find Top-K most similar for each mentor
            top_k_indices = np.argsort(similarity_matrix, axis=1) # sort based on similarity
            top_k_indices = top_k_indices[:, :k] # (N_mentors, k)
            assert top_k_indices.shape == (n_mentors, k)

            avg_scores = []
            for i in range(n_mentors):
                 # At ith mentor, get score of top mentees
                 scores = similarity_matrix[i, top_k_indices[i, :]]
                 assert len(scores) == k
                 avg_scores.append(scores.mean())
            avg_scores = np.mean(avg_scores)
        
        # TODO hungarian topk