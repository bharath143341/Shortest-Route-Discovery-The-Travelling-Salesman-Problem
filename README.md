# Shortest Route Discovery: The Travelling Salesman Problem

## Overview
This project focuses on solving the **Travelling Salesman Problem (TSP)** using **Model-Free Reinforcement Learning (MFRL)** algorithms. TSP is a classic optimization problem where the objective is to find the shortest possible route that visits a set of cities exactly once and returns to the starting point.

### Team Members
- **Sharath Chandra Kamuni**
- **Kavya Rampalli**
- **Bharath Sudha Chandra Bachala**

**Under the Guidance of:** Prof. Shivanjali Khare

---

## Objective
To solve TSP using Model-Free Reinforcement Learning (MFRL), which:
- Utilizes experience to directly learn state-action values or policies.
- Scales effectively and offers flexibility.
- Employs a trial-and-error approach to find optimal policies without requiring an explicit world model.

---

## Methodology
The solution involves two MFRL algorithms:

### Q-Learning Algorithm
- A value-based method that iteratively updates state-action values (Q-values) to find the optimal policy.
- Relies on exploration and exploitation to balance learning.
- Suitable for discrete action spaces.

### A3C (Asynchronous Advantage Actor-Critic) Algorithm
- A policy-based method that uses parallel agents to improve learning efficiency.
- Combines value-based and policy-based approaches for better convergence.
- Leverages an advantage function to reduce variance in policy updates.

---

## Problem Definition
### Global Declarations:
To define and solve the TSP, the following components are declared:
- **Environment:** Defines the cities and distances between them.
- **States:** Represent the current city and visited cities.
- **Actions:** Choices of the next city to visit.
- **Agent:** Learns the optimal route through experience.
- **Value Function:** Quantifies the quality of a state or state-action pair.
- **Reward:** Provides feedback on the quality of a chosen route.

---

## Outputs
- Both Q-Learning and A3C algorithms demonstrate convergence.
- **Q-Learning:** Shows lower initial loss, making it faster to stabilize in the early stages.
- **A3C:** Benefits from parallelism, leading to faster overall convergence.

---

## Conclusion
The project successfully solves the Travelling Salesman Problem using Model-Free Reinforcement Learning algorithms. While both Q-Learning and A3C converge, their performance varies based on initial conditions and computational resources.

---

## How to Run the Project
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Code:**
   ```bash
   python TSP.py
   ```

---


## Acknowledgments
Special thanks to Prof. Shivanjali Khare for her guidance throughout the project.
