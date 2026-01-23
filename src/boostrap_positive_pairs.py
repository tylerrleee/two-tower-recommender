import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Literal

def compute_avg_similarity(n_mentors: int,
                           similarity_matrix: npt.NDArray,
                           top_k_indices: npt.NDArray) -> float:
    """
    Helper Function: Calculates the global average similarity for the selected top-K pairings.
    """
    avg_scores = []
    
    for i in range(n_mentors):
        # Retrieve the similarity scores for only the K mentees assigned to mentor i
        # top_k_indices[i] contains the k indices for mentor i
        mentor_k_scores = similarity_matrix[i, top_k_indices[i]]
        
        # Mean similarity for this specific mentor's top-k group
        avg_scores.append(mentor_k_scores.mean())
    
    # Return the aggregate average across all mentors
    return float(np.mean(avg_scores))


def bootstrap_topk_pairs_from_embeddings(   mentor_embeddings: npt.NDArray[np.float32],
                                            mentee_embeddings: npt.NDArray[np.float32],
                                            k: int = 3,
                                            method: Literal['topk', 'hungarian_topk'] = 'hungarian_topk'
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
            top_k_indices = np.argsort(-similarity_matrix, axis=1) # sort based on similarity
            top_k_indices = top_k_indices[:, :k] # (N_mentors, k)
            assert top_k_indices.shape == (n_mentors, k)

            avg_score = compute_avg_similarity(n_mentors=n_mentors,
                                   similarity_matrix=similarity_matrix,
                                   top_k_indices=top_k_indices)
        
        elif method == 'hungarian_topk':
             # (mentor indices), (their global best match indices)
            row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)

            top_k_indices = np.zeros(shape=(n_mentors, k), dtype=np.int64)
        
            for i in range(n_mentors):
                best_match = col_ind[i]
                top_k_indices[i, 0] = col_ind[i]

                sorted_indices = np.argsort(-similarity_matrix[i])
                candidates = [indx for indx in sorted_indices if indx != best_match]
                top_k_indices[i, 1:] = candidates[:k-1]

            avg_score = compute_avg_similarity(n_mentors=n_mentors,
                                   similarity_matrix=similarity_matrix,
                                   top_k_indices=top_k_indices)
            
        else:
             raise ValueError(f"Unknown method: {method}")
        
        used_mentees = set(top_k_indices.flatten())
        all_mentees = set(range(n_mentees))
        unused_mentees = np.array(sorted(all_mentees - used_mentees), dtype=np.int64)

        print(f"Bootstrapped {k} positives per mentor with avg similarity: {avg_score:.3f}")
        print(f"  Used mentees: {len(used_mentees)} / {n_mentees}")
        print(f"  Unused mentees: {len(unused_mentees)} (will be used as hard negatives)")
        
        return top_k_indices, unused_mentees
            

            