import numpy as np
from scipy.optimize import linear_sum_assignment

def bootstrap_positive_pairs_from_embeddings(
                        mentor_embeddings: np.ndarray,
                        mentee_embeddings: np.ndarray,
                        top_k: int = 5,
                        method: str = 'hungarian'
                    ) -> np.ndarray:
        """
        Use raw S-BERT + meta embeddings to find initial positive pairs
        
        Bootstrapping strategy: we use the raw embeddings (before 
        two-tower training) to make initial pseudo-labels for positive pairs.
        S-BERT embeddings already capture semantic similarity from text

        Args:
            mentor_embeddings: (N, D) raw combined embeddings
            mentee_embeddings: (N, D) raw combined embeddings
            top_k: Number of candidates to consider
            method: 'hungarian' (global) or 'greedy' (fast)
        
        Returns:
            positive_pairs: (N,) array where positive_pairs[i] = best mentee for mentor i
        
        """
        
        # Compute cosine similarity (embeddings should be L2-normalized)
        similarity_matrix = np.dot(mentor_embeddings, mentee_embeddings.T)
        
        if method == 'hungarian':
            # Global optimizationt: Find the best mentee for each mentor (1:1)
            row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
            positive_pairs = col_ind
            avg_score = similarity_matrix[row_ind, col_ind].mean()
            
        elif method == 'greedy':
            # Greedy: for each mentor, pick best available mentee
            positive_pairs = np.zeros(len(mentor_embeddings), dtype=int)
            used_mentees = set()
            
            # Sort mentors by their max similarity (hardest first)
            mentor_order = np.argsort(-similarity_matrix.max(axis=1))
            
            for mentor_idx in mentor_order:
                # Get available mentees
                available = [i for i in range(len(mentee_embeddings)) if i not in used_mentees]
                
                # Pick best available
                scores = similarity_matrix[mentor_idx, available]
                best_local_idx = np.argmax(scores)
                best_mentee_idx = available[best_local_idx]
                
                positive_pairs[mentor_idx] = best_mentee_idx
                used_mentees.add(best_mentee_idx)
                
            avg_score = similarity_matrix[np.arange(len(positive_pairs)), positive_pairs].mean()
        
        print(f"Bootstrapped pairs with avg similarity: {avg_score:.3f}")
        return positive_pairs