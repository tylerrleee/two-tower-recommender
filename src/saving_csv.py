import os
import json
from datetime import datetime
import csv
import numpy as np

def _save_groups_summary_csv(self, output_path: str) -> None:
    """Save high-level summary of each group."""
    
    filepath = os.path.join(output_path, "groups_summary.csv")
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Group ID',
            'Mentor Name',
            'Mentor Major',
            'Mentor Email',
            'Num Mentees',
            'Avg Compatibility Score',
            'Mentee Names'
        ])
        
        # Data rows
        for result in self.results:
            mentee_names = ', '.join([m['name'] for m in result['mentees']])
            
            writer.writerow([
                result['group_id'],
                result['mentor']['name'],
                result['mentor']['major'],
                result['mentor']['email'],
                len(result['mentees']),
                f"{result['compatibility_score']:.4f}",
                mentee_names
            ])


def _save_detailed_matches_csv(self, output_path: str) -> None:
    """Save flattened mentor-mentee pairs with individual scores."""
    import csv
    
    filepath = os.path.join(output_path, "detailed_matches.csv")
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Group ID',
            'Mentor Name',
            'Mentor Major',
            'Mentor Email',
            'Mentee Name',
            'Mentee Major',
            'Mentee Year',
            'Individual Score',
            'Group Avg Score'
        ])
        
        # Data rows - one row per mentor-mentee pair
        for result in self.results:
            mentor = result['mentor']
            group_id = result['group_id']
            group_score = result['compatibility_score']
            
            for mentee, score in zip(result['mentees'], result['individual_scores']):
                writer.writerow([
                    group_id,
                    mentor['name'],
                    mentor['major'],
                    mentor['email'],
                    mentee['name'],
                    mentee['major'],
                    mentee['year'],
                    f"{score:.4f}",
                    f"{group_score:.4f}"
                ])


def _save_results_json(self, output_path: str) -> None:
    """Save complete structured results as JSON."""
    filepath = os.path.join(output_path, "results.json")
    
    # Create output structure with metadata
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_groups': len(self.results),
            'total_mentors': len(self.df_mentors) if self.df_mentors is not None else 0,
            'total_mentees': len(self.df_mentees) if self.df_mentees is not None else 0,
            'model_used': 'TwoTowerModel' if self.mentor_embeddings_learned is not None else 'Direct Matching',
            'avg_group_score': float(np.mean([r['compatibility_score'] for r in self.results]))
        },
        'groups': self.results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def _save_readable_report(self, output_path: str) -> None:
    """Save human-readable text report."""
    filepath = os.path.join(output_path, "readable_report.txt")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("MENTOR-MENTEE GROUP MATCHING RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Groups: {len(self.results)}\n")
        
        if self.results:
            avg_score = np.mean([r['compatibility_score'] for r in self.results])
            f.write(f"Average Compatibility Score: {avg_score:.4f}\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Individual group details
        for i, result in enumerate(self.results, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"GROUP {i} | Compatibility Score: {result['compatibility_score']:.4f}\n")
            f.write(f"{'='*80}\n\n")
            
            # Mentor info
            f.write(f"MENTOR:\n")
            f.write(f"  Name:  {result['mentor']['name']}\n")
            f.write(f"  Major: {result['mentor']['major']}\n")
            f.write(f"  Email: {result['mentor']['email']}\n\n")
            
            # Mentees info
            f.write(f"MENTEES ({len(result['mentees'])}):\n")
            for j, (mentee, score) in enumerate(zip(result['mentees'], result['individual_scores']), 1):
                f.write(f"  {j}. {mentee['name']}\n")
                f.write(f"     Major: {mentee['major']}\n")
                f.write(f"     Year:  {mentee['year']}\n")
                f.write(f"     Compatibility Score: {score:.4f}\n")
                if j < len(result['mentees']):
                    f.write("\n")
            
            f.write("\n")
        
        # Summary statistics at the end
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        scores = [r['compatibility_score'] for r in self.results]
        f.write(f"Total Groups: {len(self.results)}\n")
        f.write(f"Average Score: {np.mean(scores):.4f}\n")
        f.write(f"Median Score:  {np.median(scores):.4f}\n")
        f.write(f"Min Score:     {np.min(scores):.4f}\n")
        f.write(f"Max Score:     {np.max(scores):.4f}\n")
        f.write(f"Std Dev:       {np.std(scores):.4f}\n")


