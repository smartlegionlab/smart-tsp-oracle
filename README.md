# Smart TSP Oracle <sup>v0.1.1</sup>

---

![GitHub top language](https://img.shields.io/github/languages/top/smartlegionlab/smart-tsp-oracle)
[![GitHub](https://img.shields.io/github/license/smartlegionlab/smart-tsp-oracle)](https://github.com/smartlegionlab/smart-tsp-oracle/blob/master/LICENSE)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/smartlegionlab/smart-tsp-oracle)](https://github.com/smartlegionlab/smart-tsp-oracle/)
[![GitHub Repo stars](https://img.shields.io/github/stars/smartlegionlab/smart-tsp-oracle?style=social)](https://github.com/smartlegionlab/smart-tsp-oracle/)
[![GitHub watchers](https://img.shields.io/github/watchers/smartlegionlab/smart-tsp-oracle?style=social)](https://github.com/smartlegionlab/smart-tsp-oracle/)
[![GitHub forks](https://img.shields.io/github/forks/smartlegionlab/smart-tsp-oracle?style=social)](https://github.com/smartlegionlab/smart-tsp-oracle/)

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

- **[Smart TSP Solver](https://github.com/smartlegionlab/smart-tsp-oracle)** - My Python library featuring advanced heuristics (`Dynamic Gravity`, `Angular Radial`) for solving *large* TSP instances where finding the exact optimum is impractical.
- **Exact TSP Solutions (TSP ORACLE):** [exact-tsp-solver](https://github.com/smartlegionlab/exact-tsp-solver) - Optimal solutions for small instances
- **Smart TSP Benchmark** - [Smart TSP Benchmark](https://github.com/smartlegionlab/smart-tsp-benchmark) is a professional algorithm testing infrastructure with customizable scenarios and detailed metrics.
- **Spatial Optimization:** Computational geometry approaches for large-scale problems
- **Heuristic Analysis:** Comparative study of modern TSP approaches

---

## 🚀 Quick Start

```bash
# Install requirements
pip install numpy numba

# Run with 20 points
python main.py -n 20 --seed 42

# Run with custom parameters
python main.py -n 15 --seed 12345
```

### Command Line Options

```bash
-n, --num-points    Number of points (3-25 recommended)
--seed              Random seed for reproducible results
```

---

## 📜 License & Disclaimer

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

- You are free to use, modify, and distribute this software.
- **However, if you modify this software and run it as a hosted service (e.g., a web app), you MUST make the full source code of your modified version available to your users under the same license.**
- The full license text can be found in the [LICENSE](https://github.com/smartlegionlab/smart-tsp-oracle/blob/master/LICENSE) file.

### ⚠️ Important Disclaimer

> **THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**
>
> *(This is a summary of the full disclaimer, which is legally binding and located in sections 15 and 16 of the AGPLv3 license).*

For commercial use that is not compatible with the AGPLv3 terms (e.g., including this software in a proprietary product without disclosing the source code), a **commercial license** is required. Please contact me at [smartlegiondev@gmail.com](mailto:smartlegiondev@gmail.com) to discuss terms.

---

## 🧠 Algorithm Overview

### Core Components

1. **Multi-Start Greedy + 2-opt**: Generates high-quality initial solution
2. **Branch and Bound**: Exact search with mathematical optimality guarantee
3. **MST Lower Bounds**: Minimum Spanning Tree for efficient pruning
4. **Adaptive Thresholding**: Dynamic search space reduction

### Mathematical Foundation

The algorithm uses Minimum Spanning Tree (MST) calculations to compute exact lower bounds, ensuring mathematical proof of optimality for the found solutions.

---

## 📊 Example Output

```bash
python main.py -n 20 --seed 123321411
```

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
Checked: 287 paths | Speed: 574/sec | Time: 00:00:00✓ found: 4176.83 (00:00:00)
   🔍 Threshold: 3884.45 (83.1%)... ✗ cut off (00:00:00)
   🏆 The optimum has been found: 4176.83

📊 RESULTS:
==================================================
Number of points: 20
Seed: 123321411
Total possible paths: so many
Checked paths: 298
Execution time: 1.01 seconds
Speed: 295 paths/sec
Greedy + 2-opt: 4676.805998
Optimal length: 4176.825110
Improvement: 499.980888 (10.691%)

Greedy way: [3, 15, 16, 11, 2, 12, 19, 5, 8, 17, 1, 4, 7, 14, 13, 10, 6, 9, 0, 18]
The optimal path: [0, 17, 8, 5, 19, 12, 2, 11, 16, 15, 3, 14, 7, 1, 4, 13, 10, 6, 9, 18]

💾 The results are saved in tsp_result_n20_seed123321411.txt
```

---

## 📋 Performance Characteristics

- **Optimal for**: 3-25 points (exact solutions)
- **Time complexity**: O(n! * 2^n) in worst case
- **Space complexity**: O(n²) for distance matrix
- **Features**: Progress tracking, result export, reproducible runs

## 🛠️ Technical Details

### Requirements
- Python 3.8+
- numpy
- numba

### Implementation Highlights
- Numba-accelerated distance matrix computation
- Union-Find data structure for MST calculations
- Adaptive thresholding for efficient pruning
- Comprehensive result logging and export

---

**Disclaimer:** Performance results shown are for clustered/random distributions. 
Results may vary based on spatial characteristics. 
Always evaluate algorithms on your specific problem domains.

## 📞 Support

For technical support or commercial licensing inquiries:
📧 **smartlegiondev@gmail.com**