# Math4AI Colab notebooks

Runnable PyTorch code for the interactive demos of *Mathematics for AI and Machine Learning* by Fuheng Wu ([math4ai.org](https://math4ai.org)). Click a badge to open the notebook in Google Colab; nothing to install.

These notebooks are generated from the book's companion snippets; edits made here are overwritten on the next build.


## Chapter 1: Vector Space and Inner Product

| Demo | What it shows | |
|---|---|---|
| [Norm Unit Balls](chapter-01/norms.ipynb) | Interactive L1, L2, Lp, L∞, Mahalanobis, and standardized Euclidean unit-ball contours. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/norms.ipynb) |
| [Vector Addition Parallelogram](chapter-01/vector-addition.ipynb) | Drag u and v; see c = u + v and coordinate projections. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/vector-addition.ipynb) |
| [Outer Product u vᵀ](chapter-01/outer-product.ipynb) | Heatmap of u vᵀ with highlighted entry uᵢ vⱼ. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/outer-product.ipynb) |
| [Inner Product and Angle Explorer](chapter-01/inner-product.ipynb) | Interactively connect inner product, norms, and angle between two vectors. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/inner-product.ipynb) |
| [Orthogonal Projection onto Span](chapter-01/projection-span.ipynb) | Project a vector onto a one-dimensional subspace and inspect residual orthogonality. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/projection-span.ipynb) |
| [Hyperplane Distance](chapter-01/hyperplane-distance.ipynb) | 3D hyperplane θᵀx = b, foot of perpendicular, and signed distance d⊥. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/hyperplane-distance.ipynb) |
| [Linear Regression Geometry](chapter-01/linear-regression.ipynb) | Hyperplane of predictions, data point, orthogonal foot, and residual in augmented space. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/linear-regression.ipynb) |
| [Weight Decay Bowl](chapter-01/weight-decay-bowl.ipynb) | Quadratic potential (λ/2)||W||², trajectory, W* vs W_opt with weight decay. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/weight-decay-bowl.ipynb) |
| [L1 vs L2 Regularization](chapter-01/l1-l2-geometry.ipynb) | Constraint geometry and lasso/ridge paths—why L1 hits axes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/l1-l2-geometry.ipynb) |
| [Cauchy-Schwarz Bound Visualizer](chapter-01/cauchy-schwarz.ipynb) | See dot-product magnitude compared with product of norms. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/cauchy-schwarz.ipynb) |
| [Gram-Schmidt Orthonormalization](chapter-01/gram-schmidt.ipynb) | Build an orthonormal basis from two input vectors step by step. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-01/gram-schmidt.ipynb) |

## Chapter 2: Matrix

| Demo | What it shows | |
|---|---|---|
| [Matrix-Vector Multiplication Views](chapter-02/matvec-views.ipynb) | Compare row-dot-product and column-linear-combination interpretations. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-02/matvec-views.ipynb) |
| [Outer Product Matrix Build](chapter-02/outer-product.ipynb) | Construct a rank-1 matrix from two vectors and inspect structure. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-02/outer-product.ipynb) |
| [Matrix Multiplication Four Views](chapter-02/matmul-views.ipynb) | Track equivalent entries from row-column, row-wise, column-wise, and outer-product views. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-02/matmul-views.ipynb) |
| [Condition Number and Sensitivity](chapter-02/condition-number.ipynb) | See how near-singular matrices amplify perturbations in linear solves. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-02/condition-number.ipynb) |

## Chapter 3: Subspaces and Orthogonality

| Demo | What it shows | |
|---|---|---|
| [Subspace Membership Checker](chapter-03/subspace-membership.ipynb) | Check if a target vector lies in the span of two basis vectors in 2D. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-03/subspace-membership.ipynb) |
| [Orthogonal Decomposition Explorer](chapter-03/orthogonal-decomposition.ipynb) | Split a vector into projection and orthogonal residual components. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-03/orthogonal-decomposition.ipynb) |
| [Least Squares Line Fit](chapter-03/least-squares-line.ipynb) | Fit y=ax+b by minimizing squared residuals and visualize errors. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-03/least-squares-line.ipynb) |
| [Null Space and Column Space View](chapter-03/null-column-space.ipynb) | Inspect rank, nullity, and geometric directions for a 2x2 matrix. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-03/null-column-space.ipynb) |
| [Rotary Position Embedding (RoPE)](chapter-03/rope.ipynb) | Rotate query/key pairs by position; the attention score depends only on the relative distance n − m. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-03/rope.ipynb) |

## Chapter 4: QR Decomposition and Numerical Rank

| Demo | What it shows | |
|---|---|---|
| [Classical Gram-Schmidt to QR](chapter-04/gs-qr.ipynb) | Compute Q and R for a 2x2 matrix and verify A=QR. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-04/gs-qr.ipynb) |
| [Householder Reflection Geometry](chapter-04/householder-geometry.ipynb) | Reflect vectors across a hyperplane and observe norm preservation. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-04/householder-geometry.ipynb) |
| [Numerical Rank via Singular Values](chapter-04/numerical-rank.ipynb) | Explore threshold-dependent effective rank from singular values. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-04/numerical-rank.ipynb) |
| [Orthogonality Loss Under Perturbation](chapter-04/orthogonality-loss.ipynb) | See how nearly collinear columns challenge numerical orthogonalization. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-04/orthogonality-loss.ipynb) |

## Chapter 5: Square Matrix and LU Decomposition

| Demo | What it shows | |
|---|---|---|
| [LU Factorization Step Viewer](chapter-05/lu-steps.ipynb) | Factor a 2x2 matrix into L and U and verify the reconstruction. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-05/lu-steps.ipynb) |
| [Pivoting Stability Explorer](chapter-05/pivoting-stability.ipynb) | Compare elimination with and without partial pivoting on near-singular systems. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-05/pivoting-stability.ipynb) |
| [Determinant from Triangular Factors](chapter-05/det-triangular.ipynb) | Connect det(A) with diagonal products of triangular factors. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-05/det-triangular.ipynb) |
| [Forward and Back Substitution](chapter-05/forward-backward.ipynb) | Solve Ly=b then Ux=y and compare with direct solve. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-05/forward-backward.ipynb) |

## Chapter 7: Symmetric Matrix

| Demo | What it shows | |
|---|---|---|
| [Spectral Clustering and the Fiedler Vector](chapter-07/spectral-clustering.ipynb) | Normalized graph Laplacian of two linked clusters; the Fiedler vector's signs split the graph. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-07/spectral-clustering.ipynb) |

## Chapter 10: Matrix Block Partitioning

| Demo | What it shows | |
|---|---|---|
| [Tiled Softmax Attention](chapter-10/tiled-softmax.ipynb) | Online softmax over key/value blocks (the FlashAttention idea) matches full attention exactly. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-10/tiled-softmax.ipynb) |

## Chapter 19: Stochastic Differential Equations

| Demo | What it shows | |
|---|---|---|
| [Reverse Diffusion: Noise to Data](chapter-19/reverse-diffusion.ipynb) | Run the reverse-time SDE (or probability flow ODE) with the exact score from noise back to a two-mode distribution. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-19/reverse-diffusion.ipynb) |

## Chapter 20: ODE, SDE, and Continuous Limits of Algorithms

| Demo | What it shows | |
|---|---|---|
| [Flow Matching](chapter-20/flow-matching.ipynb) | Straight training paths, curved non-crossing flow: the exact flow matching velocity carries noise to a 3-mode target. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-20/flow-matching.ipynb) |
