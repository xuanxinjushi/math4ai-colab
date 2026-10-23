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

## Chapter 6: Geometric Transformation and Eigen Decomposition

| Demo | What it shows | |
|---|---|---|
| [Linear Transform of Unit Circle](chapter-06/unit-circle-transform.ipynb) | Map the unit circle by a 2x2 matrix and inspect stretching and rotation. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-06/unit-circle-transform.ipynb) |
| [Eigenvector Direction Check](chapter-06/eigenvector-check.ipynb) | Test candidate directions and see when Av stays collinear with v. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-06/eigenvector-check.ipynb) |
| [Power Iteration Convergence](chapter-06/power-iteration.ipynb) | Iterate x_{k+1}=Ax_k/||Ax_k|| and track dominant eigenvector convergence. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-06/power-iteration.ipynb) |
| [Diagonalization Recomposition](chapter-06/diagonalization.ipynb) | Rebuild A as PDP^{-1} and inspect reconstruction error. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-06/diagonalization.ipynb) |

## Chapter 7: Symmetric Matrix

| Demo | What it shows | |
|---|---|---|
| [Quadratic Form Geometry](chapter-07/quadratic-form.ipynb) | Visualize x^T A x level behavior for 2x2 symmetric matrices. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-07/quadratic-form.ipynb) |
| [Spectral Theorem Explorer](chapter-07/spectral-theorem.ipynb) | Inspect orthonormal eigenvectors and diagonalization of symmetric matrices. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-07/spectral-theorem.ipynb) |
| [Positive Definiteness Check](chapter-07/positive-definite.ipynb) | Track eigenvalue and principal-minor conditions for SPD behavior. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-07/positive-definite.ipynb) |
| [Rayleigh Quotient Bounds](chapter-07/rayleigh-bounds.ipynb) | See how x^T A x / x^T x stays between minimum and maximum eigenvalues. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-07/rayleigh-bounds.ipynb) |
| [Spectral Clustering and the Fiedler Vector](chapter-07/spectral-clustering.ipynb) | Normalized graph Laplacian of two linked clusters; the Fiedler vector's signs split the graph. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-07/spectral-clustering.ipynb) |

## Chapter 8: Singular Value Decomposition (SVD)

| Demo | What it shows | |
|---|---|---|
| [SVD Geometry of Unit Circle](chapter-08/svd-geometry.ipynb) | Decompose a 2x2 map into rotation-stretch-rotation and track image axes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-08/svd-geometry.ipynb) |
| [Low-Rank Approximation Error](chapter-08/low-rank-error.ipynb) | Compare Frobenius error of rank-k truncations from singular values. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-08/low-rank-error.ipynb) |
| [Pseudo-Inverse and Least Squares](chapter-08/pseudoinverse-ls.ipynb) | Solve overdetermined systems via A^+ and visualize residuals. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-08/pseudoinverse-ls.ipynb) |
| [Conditioning Through Singular Values](chapter-08/singular-conditioning.ipynb) | Inspect condition number and sensitivity as smallest singular value changes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-08/singular-conditioning.ipynb) |

## Chapter 9: Matrix Decompositions and Beyond

| Demo | What it shows | |
|---|---|---|
| [Polar Decomposition Explorer](chapter-09/polar-decomposition.ipynb) | Split a matrix into orthogonal and symmetric factors and inspect reconstruction. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-09/polar-decomposition.ipynb) |
| [Schur Form Intuition](chapter-09/schur-form.ipynb) | Relate eigenvalues to triangular Schur form entries for 2x2 examples. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-09/schur-form.ipynb) |
| [Jordan Sensitivity Toy Model](chapter-09/jordan-sensitivity.ipynb) | Show how nearly defective matrices react strongly to small perturbations. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-09/jordan-sensitivity.ipynb) |
| [Matrix Function via Eigenvalues](chapter-09/matrix-function-eigs.ipynb) | Compare f(A) from spectral mapping with direct series-style intuition. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-09/matrix-function-eigs.ipynb) |

## Chapter 10: Matrix Block Partitioning

| Demo | What it shows | |
|---|---|---|
| [Block Matrix Multiplication](chapter-10/block-multiplication.ipynb) | Verify block-product rules against elementwise multiplication results. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-10/block-multiplication.ipynb) |
| [Schur Complement Lens](chapter-10/schur-complement.ipynb) | Inspect how eliminating one block yields Schur complement updates. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-10/schur-complement.ipynb) |
| [Block Gaussian Elimination](chapter-10/block-elimination.ipynb) | Follow elimination steps on a 2x2 block system and track residuals. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-10/block-elimination.ipynb) |
| [Woodbury Identity Playground](chapter-10/woodbury.ipynb) | Compare direct inverse update with Woodbury low-rank formula. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-10/woodbury.ipynb) |
| [Tiled Softmax Attention](chapter-10/tiled-softmax.ipynb) | Online softmax over key/value blocks (the FlashAttention idea) matches full attention exactly. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-10/tiled-softmax.ipynb) |

## Chapter 11: Matrix Calculus

| Demo | What it shows | |
|---|---|---|
| [Gradient Field and Descent Direction](chapter-11/gradient-field.ipynb) | Visualize gradients as vectors and compare with steepest-descent steps. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-11/gradient-field.ipynb) |
| [Jacobian Chain Rule Explorer](chapter-11/jacobian-chain.ipynb) | Compose two vector maps and compare direct Jacobian with chain-rule product. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-11/jacobian-chain.ipynb) |
| [Hessian Curvature Lens](chapter-11/hessian-curvature.ipynb) | Inspect eigenvalues of Hessian to classify local curvature and conditioning. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-11/hessian-curvature.ipynb) |
| [Second-Order Taylor Approximation](chapter-11/taylor-second-order.ipynb) | Compare exact function values with local second-order Taylor expansion around a chosen point. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-11/taylor-second-order.ipynb) |

## Chapter 12: Optimization Methods

| Demo | What it shows | |
|---|---|---|
| [Gradient Descent Step Size Explorer](chapter-12/gd-stepsize.ipynb) | Inspect convergence or divergence of gradient descent as learning rate changes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-12/gd-stepsize.ipynb) |
| [Newton vs Gradient Descent](chapter-12/newton-vs-gd.ipynb) | Compare first-order and second-order updates on a curved one-dimensional objective. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-12/newton-vs-gd.ipynb) |
| [Projected Gradient on Box Constraint](chapter-12/projected-gradient.ipynb) | Show unconstrained steps followed by projection onto a feasible interval. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-12/projected-gradient.ipynb) |
| [Duality Gap Toy Problem](chapter-12/duality-gap.ipynb) | Visualize primal and dual objectives and the effect of strong duality conditions. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-12/duality-gap.ipynb) |

## Chapter 19: Stochastic Differential Equations

| Demo | What it shows | |
|---|---|---|
| [Reverse Diffusion: Noise to Data](chapter-19/reverse-diffusion.ipynb) | Run the reverse-time SDE (or probability flow ODE) with the exact score from noise back to a two-mode distribution. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-19/reverse-diffusion.ipynb) |

## Chapter 20: ODE, SDE, and Continuous Limits of Algorithms

| Demo | What it shows | |
|---|---|---|
| [Flow Matching](chapter-20/flow-matching.ipynb) | Straight training paths, curved non-crossing flow: the exact flow matching velocity carries noise to a 3-mode target. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/math4ai-colab/blob/main/chapter-20/flow-matching.ipynb) |
