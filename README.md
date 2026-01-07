
# Mentor-Mentee Matching System
## Deep Learning-Based Big/Little Pairing for University Programs

This system uses a Two-Tower Neural Network architecture with contrastive learning to match mentors ("Bigs") with mentees ("Littles") in university mentorship programs. By combining semantic text understanding (S-BERT), structured profile features, and objective optimization, the system generates mentor-to-mentee groups that balance compatibility and diversity.

## Reason for this Project

Traditional approaches to match 100-200 applications rely on manual review or simple keyword matching. This system aims to:

- Learns semantic similarity from profile text using pre-trained language models (S-BERT)
- Optimizes globally to maximize overall match 
- Encourages diversity within mentee pairs to prevent echo chambers, or groups of overfitting personality (inability to learn new hobbies or experience from each other)
- Scales efficiently to handle thoudsands of applications 
- Learned objective to reduce inconsistent criteria where applications are not the same every cycle, and with different leaderships
- Automate matching to reduce time intensive manual matching (which usually takes 2 weeks between 2 coordinator for 200 applicants)

## Similar Works

Similar works for Big-Little in fraternities and sororities have been done many times. However, the downside to the Greek life point of view is the scale of matching, where popular matching algorithms like Genetic & Stable Marriage relies on labeled preferences for all candidates for each respective role. Hence, having 100+ applicants is unrealistic to have an applicant rank everyone, when they have never met, know or have the time to rank everyone themselves. 

The *Stable Marriage Problem* requires every applicant to rank every other applicant. For a program with 300 mentors and 600 mentees:

- Each mentor would need to rank 600 mentees
- Each mentee would need to rank 300 mentors
- Total rankings required: 360,000 

The approach: Use content-based features to compute compatibility automatically, requiring zero manual rankings 

## Core Components

1. Feature Engineering

2. Embedding Generation

3. Two-Tower model

4. Diversity Aware Loss

5. Matching Algorithm

6. Training Pipeline

## Why Two-Tower Architecture over other Approaches?

## Diversity Loss Equation

$$
\[
\mathcal{L}_{\text{total}}
=
\lambda_c
\left(
- \frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{
\exp\left(\frac{\mathbf{m}_i^\top \mathbf{e}_i}{\tau}\right)
}{
\sum_{j=1}^{N}
\exp\left(\frac{\mathbf{m}_i^\top \mathbf{e}_j}{\tau}\right)
}
\right)
+
\lambda_d
\left(
\frac{1}{N(N-1)}
\sum_{i \neq j}
\max\left(0, \cos(\mathbf{d}_i, \mathbf{d}_j) - \delta \right)
\right)
\]
$$

<a href="https://github.com/haley/hatchamatch?tab=readme-ov-file"> Hatch-a-Match - Genetic Algorithm</a>

<a href="https://www.math.cmu.edu/users/math/af1p/Teaching/OR2/Projects/P44/E.pdf"> Big-Little Pairings at Carnegie Mellon Univeristy - Stable Marriage Problem & Greedy</a>

# Initial Challenges 

Big/Little and Mentor/Mentee programs are different for different organizations, in terms of:

1. Volumn of applications (100-300+)
2. Type of questions being asked 
3. Preferences in a little/bigs

what we know that are the same:

1. Most programs are surveyed through Google Form, so the corpus we are working with are:

- Short and long responses to open-ended questions
- Preference scale from 0-5
- Big or Little

## Which means

1. The volumn of application means we cannot utilize the *Stable Marriage Problem*, as it is unrealistic to have every applicant rank every other applicant

2. Cold Start Problem: most applicants does not prior history or documented context with the organization

2. Transfer Learning: Making a model that can work for future applications, even if the questions changes. 

3. Speed: Every semester, Recommendations are made once, so speed is not necessarily the priority

# Resources used

<a href="https://peps.python.org/pep-0008/"> Python Coding Style</a>

<a href="https://docs.python.org/2/library/unittest"> Unit Tests </a>

<a href="https://developers.google.com/machine-learning/recommendation"> Google Recommendation Systems </a>

<a href="https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture"> Google Cloud Tensorflow deep retrieval using Two Towers architecture </a>


