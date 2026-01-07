import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations
import faiss

class GroupMatcher:
    """
    Match 1 mentor w/ 2 mentees
    Problem solving:
    - Stable Marriage Problem
    - Hungarian Algo
    """
    def __init__(
        self,
        model,
        faiss_index = None,
        compatibility_weight=0.6,
        diversity_weight=0.4
    ):
        self.model = model
        self.compatibility_weight = compatibility_weight
        self.diversity_weight = diversity_weight

    def compute_compatability(self, mentor_emb, mentee_emb):
        """ Cosine Similarity between mentor and mentees"""
        return np.dot(mentor_emb, mentee_emb.T)
    
    def compute_diversity(self, mentee1_features, mentee2_features):
        """
        Measure diversity between two mentees
        higher score = more diverse
        """
        # Compute average difference
        diff = np.abs(mentee1_features - mentee2_features)
        return np.mean(diff)
    
    def _format_results(self, mentor_indices, mentee_indices, cost_matrix):
        """ Helper to convert indices into final group dictionary.
        Standardize formatting
            Return: 
                A dictionary of {mentor_index : mentee index}
        """
        groups = {}
        for slot_index, mentee_index in zip(mentor_indices, mentee_indices):
            real_mentor_index = slot_index // 2

            match_score = cost_matrix[slot_index, mentee_index]

            if real_mentor_index not in groups:
                groups[real_mentor_index] = {
                    'mentees': [],
                    'individual_scores': [],
                    'total_compatibility_score': 0
                }
            groups[real_mentor_index]['mentees'].append(int(mentee_index))
            groups[real_mentor_index]['individual_scores'].append(float(match_score))
            groups[real_mentor_index]['total_compatibility_score'] += float(match_score)
        return groups
    

    def find_best_groups_base(self, mentor_emb, mentee_emb):
        """
        Find Optimal group of 1 mentor + 2 mentees
        1. Duplicate mentor embedding so that each mentor gets "2 slots"
        2. Compute cost matrix for all possible pairings
        3. Negates scores by maximizing similarity globally
        
        Complexity: O(n^3)
        Bias: Good for smaller dataset
        Returns:
            groups: List of (mentor_index, [mentee_index1, mentee_index2])
        """

        n_mentors = len(mentor_emb)
        n_mentees = len(mentee_emb)

        # Duplicate mentors sos each has 2 'slots' for mentees
        expanded_mentor_emb = np.repeat(mentor_emb, 2, axis=0)

        cost_matrix = np.dot(expanded_mentor_emb, mentee_emb.T)
        
        # Find optimal Assignment
        mentor_indices, mentee_indices = linear_sum_assignment(cost_matrix, maximize = True)
        return self._format_results(mentor_indices, mentee_indices, cost_matrix)

    def find_best_groups_faiss(self, mentor_emb, mentee_emb, top_k=10):
        """
        Retrieve Augmented Matching: Uses FAISS to find Top-K mentors for each
        mentee, optimizes matching after. 
        Current implementation is CPU

        1. Builds index for mentor embeddings
        2. For each mentee, retrieve top-k mentor candidates
        3. Creates cost matrix only for candidates
        
        Goal: Sacrifise precision for efficiency = near-optimal
        Bias: for Large Datasets at O(k*n*log(m))

        """

        if self.faiss_index is None:
            # Build index if not provided
            dim = mentor_emb.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(mentor_emb.astype('float32'))
        
        # 1. Retrieval : get top_k mentor for every mentee
        # distances = (n_mentees, top_k), indcices = (n_mentees, top_k)
        distances, mentor_candidate_indices = self.faiss_index.search(mentee_emb.astype('float32')
                                                                      , top_k)
        # 2. Sparse Matrix : Cost matrix for top_k candidates
        # low starting score to discourage matching non-candidate
        n_mentors = len(mentor_emb)
        n_mentees = len(mentee_emb)
        sparse_cost_matrix = np.full((n_mentors * 2, n_mentees), -1e9)

        for mentee_index in range(n_mentees):
            for rank in range(top_k):
                # Get mentor candidate index & their score
                m_index = mentor_candidate_indices[mentee_index, rank]
                score = distances[mentee_index, rank]

                # Fill cost matrix with mentor
                sparse_cost_matrix[m_index * 2, mentee_index] = score
                sparse_cost_matrix[m_index * 2 + 1, mentee_index] = score
        
        # 3. Optimization step
        mentor_indices, mentee_indices = linear_sum_assignment(sparse_cost_matrix, maximize=True)
        
        return self._format_results(mentor_indices=mentee_indices, 
                                    mentee_indices=mentee_indices, 
                                    cost_matrix=sparse_cost_matrix)



