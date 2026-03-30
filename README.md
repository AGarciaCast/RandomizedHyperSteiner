# HyperbolicSteinerTrees
This folder contains the code for the paper called **"Randomized HyperSteiner: A Stochastic Delaunay Triangulation Heuristic for the Hyperbolic Steiner Minimal Tree"**.

# Abstract
We study the problem of constructing Steiner Minimal Trees (SMTs) in hyperbolic space. Exact SMT computation is NP-hard, and existing hyperbolic heuristics such as HyperSteiner are deterministic and often get trapped in locally suboptimal configurations. We introduce **Randomized HyperSteiner** (RHS), a stochastic Delaunay triangulation heuristic that incorporates randomness into the expansion process and refines candidate trees via Riemannian gradient descent optimization. Experiments on synthetic data sets and a real-world single-cell transcriptomic data show that RHS outperforms Minimum Spanning Tree (MST), Neighbour Joining, and vanilla HyperSteiner (HS). In near-boundary configurations, RHS can achieve a 32% reduction in total length over HS, demonstrating its effectiveness and robustness in diverse data regimes.

# Information
- The code for the Randomized HyperSteiner method is in the exhaustiveMethod_global function in the following file: `src/exhaustiveMethod_global.py`. <br />
- The experiments can be reproduced using `syntheticExperiments.ipynb` for *Centered Gaussian*, *Mixture of Gaussians Near Boundary*, and *Approaching the Theoretical Limit; `realExperiments.ipynb` for *Real Biological Data*; and `color_map_exp.py` for *Characterizing the Transition Zone*. <br />
- The folder `Data` contains the hyperbolic representation of the Planaria dataset in the Klein-Beltrami disk. This is used for the *Real Biological Data* experiment.


# Setup

To install the requirements, we use conda. We recommend creating a new environment for the project.
```
conda create --name "hyper" python=3.8
conda activate hyper
```

Install the relevant dependencies.
```
./setup_phcp.sh 
pip install -r requirements.txt
conda install -c conda-forge biotite==0.35.0
```