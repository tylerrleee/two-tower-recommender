import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations

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
    
    def find_best_groups(
            self,
            mentor_emb,
            mentee_emb,
            mentee_diversity,
            n_mentees_per_mentor = 2
    ):
        """
        Find Optimal group of 1 mentor + 2 mentees

        Returns:
            groups: List of (mentor_index, [mentee_index1, mentee_index2])
        """

        n_mentors = len(mentor_emb)
        n_mentees = len(mentee_emb)

        # Duplicate mentors sos each has 2 'slots' for mentees
        expanded_mentor_emb = np.repeat(mentor_emb, 2, axis=0)

        cost_matrix = np.dot(expanded_mentor_emb, mentee_emb.T)
        
        # Find optimal Assignment
        mentor_indices, pair_indices = linear_sum_assignment(cost_matrix, maximize = True)

        groups = {}
        for slot_index, mentee_idex in zip(mentor_indices, pair_indices):
            real_mentor_index = slot_index // 2 # Convert slots back to [0, 0, 1,1, ...]
            if real_mentor_index not in groups:
                groups[real_mentor_index] = []
            groups[real_mentor_index].append(mentee_idex)

        return groups




