# Smart TSP Oracle <sup>v0.0.1</sup>

---

A high-performance, exact solver for the Traveling Salesman Problem (TSP) implemented in Python. Utilizes an intelligent Branch and Bound algorithm with adaptive thresholding to find the globally optimal solution for small to medium-sized TSP instances.

## Go version: [exact-tsp-solver](https://github.com/smartlegionlab/exact-tsp-solver)

---

## 👨‍💻 Author

[**A.A. Suvorov**](https://github.com/smartlegionlab/)

*   Passionate about pushing the boundaries of algorithmic optimization.
*   This solver was developed to bridge the gap between theoretical computer science and practical implementation.

- Researcher specializing in computational optimization and high-performance algorithms
- Focused on bridging theoretical computer science with practical engineering applications
- This project represents extensive research into spatial optimization techniques

## 🔗 Related Research

For those interested in the theoretical foundations:

- **[Smart TSP Solver](https://github.com/smartlegionlab/smart-tsp-solver)** - My Python library featuring advanced heuristics (`Dynamic Gravity`, `Angular Radial`) for solving *large* TSP instances where finding the exact optimum is impractical.
- **Exact TSP Solutions (TSP ORACLE):** [exact-tsp-solver](https://github.com/smartlegionlab/exact-tsp-solver) - Optimal solutions for small instances
- **Smart TSP Benchmark** - [Smart TSP Benchmark](https://github.com/smartlegionlab/smart-tsp-benchmark)  is a professional algorithm testing infrastructure with customizable scenarios and detailed metrics.
- **Spatial Optimization:** Computational geometry approaches for large-scale problems
- **Heuristic Analysis:** Comparative study of modern TSP approaches

---

**Disclaimer:** Performance results shown are for clustered/random distributions. 
Results may vary based on spatial characteristics. 
Always evaluate algorithms on your specific problem domains.

---

## 📜 Licensing

This project is offered under a dual-licensing model.

### 🆓 Option 1: BSD 3-Clause License (for Non-Commercial Use)
This license is **free of charge** and allows you to use the software for:
- Personal and educational purposes
- Academic research and open-source projects
- Evaluation and testing

**Important:** Any use by a commercial organization or for commercial purposes (including internal development and prototyping) requires a commercial license.

### 💼 Option 2: Commercial License (for Commercial Use)
A commercial license is **required** for:
- Integrating this software into proprietary products
- Using it in internal operations within a company
- SaaS and hosted services that incorporate this software

**Important:** The commercial license provides usage rights but **does not include any indemnification or liability**. The software is provided "AS IS" without any warranties as described in the full license agreement.

**To obtain a commercial license,** please contact us directly at:  
📧 **smartlegiondev@gmail.com**

---

## Output

`python main.py -n 20 --seed 123321411`

```
==================================================
🚀 TSP SOLVER (ORACLE v2) - 20 POINTS
🔢 SEED: 123321411
==================================================

📍 Coordinates of points:
   Dot 0: (716.02, 797.47)
   Dot 1: (336.04, 587.85)
   Dot 2: (620.65, 170.96)
   Dot 3: (0.77, 335.69)
   Dot 4: (275.75, 747.51)
   Dot 5: (823.69, 50.79)
   Dot 6: (533.09, 748.58)
   Dot 7: (135.94, 668.09)
   Dot 8: (890.62, 294.76)
   Dot 9: (583.21, 863.07)
   Dot 10: (463.36, 816.27)
   Dot 11: (592.15, 238.06)
   Dot 12: (680.65, 113.63)
   Dot 13: (404.13, 996.27)
   Dot 14: (14.42, 824.82)
   Dot 15: (87.71, 239.60)
   Dot 16: (443.56, 190.80)
   Dot 17: (724.89, 425.34)
   Dot 18: (929.70, 824.26)
   Dot 19: (726.95, 159.40)
1. Launching the multi-start greedy algorithm...
   ✅ Multi-start greedy + 2-opt: length = 4676.81
   🎯 We start the search from 4209.13 (90.0%)
Checked: 288 paths | Speed: 563/sec | Time: 00:00:00✓ found: 4176.83 (00:00:00)
   🔍 Threshold: 3884.45 (83.1%)... ✗ cut off (00:00:00)
   🏆 The optimum has been found: 4176.83

📊 RESULTS:
==================================================
Number of points: 20
Seed: 123321411
Total possible paths: so many
Checked paths: 0
Execution time: 1.00 seconds
Speed: 0 paths/sec
Greedy + 2-opt: 4676.805998
Optimal length: 4176.825110
Improvement: 499.980888 (10.691%)

Greedy way: [3, 15, 16, 11, 2, 12, 19, 5, 8, 17, 1, 4, 7, 14, 13, 10, 6, 9, 0, 18]
The optimal path: [0, 17, 8, 5, 19, 12, 2, 11, 16, 15, 3, 14, 7, 1, 4, 13, 10, 6, 9, 18]

💾 The results are saved in tsp_result_n20_seed123321411.txt
```