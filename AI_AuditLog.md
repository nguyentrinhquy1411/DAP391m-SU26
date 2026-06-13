# AI Audit Log - Metadata & Summary

## Student Information
* **Student Name**: Nguyễn Trinh Quý
* **Student ID**: SE203691
* **Course**: DAP391m (RBL Insight Framework - AI Reflection 30%)
* **Assignment**: Active Evidential Sensing and Risk-Averse Rescue Routing (AES-RARR) for UAV-Enabled Maritime Search and Rescue: A Simulation Study

---

## AI Usage Summary
* **Total Prompts Used (all AI tools)**: ~150
* **Core Prompts Logged**: 19
* **Selection Ratio**: 12.7% (19 / 150 - falls within the required 10-20% range)
* **Hallucination Cases Detected**: 3 cases (meets the $\ge 3$ cases requirement)

---

## AI Tools Used
| AI Tool | Purpose | Frequency | Main Value |
| :--- | :--- | :--- | :--- |
| **ChatGPT (GPT-4o)** | Code generation, quick debugging, mathematical formatting | High | Rapid prototyping, template generation |
| **Claude (Sonnet 3.5)** | Requirements analysis, structural refactoring, paper review, design critiques | Medium | Deep logical reasoning, high-fidelity LaTeX drafting |
| **GitHub Copilot** | Inline code autocompletion | High | Accelerated repetitive coding and boilerplate generation |

---

## Core Prompts Distribution
| Component / Stage | Number of Prompts | Required (Min) |
| :--- | :---: | :---: |
| **Business & Problem Understanding** | 3 | $\ge 2$ |
| **Data Understanding & Preparation** | 4 | $\ge 3$ |
| **Exploratory Data Analysis (EDA)** | 3 | $\ge 2$ |
| **Modeling & Regression Analysis** | 5 | $\ge 4$ |
| **Evaluation, Visualization & Reporting** | 4 | $\ge 3$ |
| **Total** | **19** | **15 - 20** |

---

# Core Prompts Details

### Entry #: 001
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Business & Problem Understanding
* **Problem/Context**: Defining the "silent drowning" failure mode under wave occlusion in maritime search and rescue.
* **Prompt to AI**: 
  > "In UAV-based maritime search and rescue, vision models output deterministic softmax probabilities. However, waves and glare cause temporary occlusion. Can you help me define a failure case where a drowning victim is deprioritized due to model uncertainty, and how we can mathematically frame this as an active sensing problem?"
* **AI Response (Summary)**: The AI defined the "silent drowning" failure mode, showing that standard softmax networks output high-entropy (flat) distributions under occlusion. Because the expected risk is calculated from these flat distributions, drowning victims are assigned lower priority scores, causing the path planner to skip them. It suggested using Evidential Deep Learning (EDL) to capture epistemic uncertainty and using this uncertainty to trigger local verification descents.
* **Human Delta & Reflection**:
  * *Critical Thinking*: The AI's definition was theoretically sound but ignored flight kinetics: if the UAV verifies everything, flight latency will increase time-to-rescue for other victims.
  * *Contextualization*: Real search-and-rescue UAVs operate under strict battery and time-to-rescue constraints.
  * *Creative Synthesis*: I modified the definition by introducing a physiological decay model to balance the value of active verification against travel time latency.
  * *Decision Ownership*: I decided to implement a dual-trigger mechanism that only descends when expected risk is high AND epistemic uncertainty is high, preventing unnecessary flight overhead.

### Entry #: 002
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Business & Problem Understanding
* **Problem/Context**: Deciding when the UAV should descend to verify targets.
* **Prompt to AI**: 
  > "We want to trigger a lower-altitude verification descent when the target is both uncertain and risky. What thresholds should we use, and what are the trade-offs of descending from patrol altitude (40m) to verification altitude (20m) in terms of latency and spatial covariance?"
* **AI Response (Summary)**: The AI proposed a dual-threshold trigger: $u > \tau_{unc}$ and $\mathbb{E}[R] > \tau_{risk}$. It derived the covariance reduction factor based on the camera pinhole model: reducing altitude by half scales the spatial covariance by $\rho^2 = (z_{verify}/z_{search})^2 = 0.25$.
* **Human Delta & Reflection**:
  * *Critical Thinking*: The AI assumed descents are instantaneous and ignored the transition flight latency.
  * *Contextualization*: Micro-UAVs have limited battery life, so descent frequency must be tightly controlled.
  * *Creative Synthesis*: I added a vertical transit latency parameter ($v_{\downarrow}$ and descent time) to penalize excessive descents in our simulation loop.
  * *Decision Ownership*: Implemented the dual-threshold triggers with a vertical flight speed of 3m/s in our simulation loop.

### Entry #: 003
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Business & Problem Understanding
* **Problem/Context**: Incorporating clutter (visual distractors) to make the simulation realistic for Q2-Q3 standards.
* **Prompt to AI**: 
  > "Reviewers will reject a maritime search and rescue model that doesn't account for visual clutter like buoys or debris. How should we model distractors in our simulation, and how can the active verification branch be used to filter them out?"
* **AI Response (Summary)**: The AI suggested adding non-victim objects with high initial expected risk and high uncertainty. Once the active branch is triggered and the UAV descends, the occlusion is resolved, and the classifier can identify them as distractors (e.g. class 4: buoy) and filter them from the rescue queue.
* **Human Delta & Reflection**:
  * *Critical Thinking*: AI forgot that if a distractor is visited, it wastes a physical rescue package.
  * *Contextualization*: Real UAVs carry limited physical assets (e.g., life jackets).
  * *Creative Synthesis*: I integrated a rescue package limit to the simulation and defined a False Alarm (FA) rate metric.
  * *Decision Ownership*: Implemented the distractor filtering logic in `simulation.py` and calculated the FA rate to demonstrate a drop from 92.0% to 43.5%.

---

### Entry #: 004
* **Prompt Type**: PROBLEM-SOLVING
* **Stage/Component**: Data Understanding & Preparation
* **Problem/Context**: Extracting image crops and annotations from SeaDronesSee.
* **Prompt to AI**: 
  > "I am using the SeaDronesSee dataset. How can I parse the COCO JSON annotations to filter classes like swimmer, floater, and life jacket, and crop corresponding 64x64 images to simulate the inputs for our Evidential CNN?"
* **AI Response (Summary)**: The AI provided a script using the `json` library to load annotations, map category IDs (1: swimmer, 2: floater, 4: life jacket) to our simulated target classes, and crop bounding boxes from the images.
* **Human Delta & Reflection**:
  * *Critical Thinking*: The AI script crashed when images were missing from the local folder.
  * *Contextualization*: Real datasets are often split across train/val/test directories.
  * *Creative Synthesis*: I wrapped the image loading in a `try/except` block and cached the crops in memory (`_crop_cache`) to avoid repeated disk reads.
  * *Decision Ownership*: Adopted this preprocessing pipeline in `src/simulation.py`.

### Entry #: 005
* **Prompt Type**: PROBLEM-SOLVING
* **Stage/Component**: Data Understanding & Preparation
* **Problem/Context**: Generating realistic deteriorated inputs to simulate wave occlusion.
* **Prompt to AI**: 
  > "In a real maritime environment, wave foam and spray blur the targets. How can we simulate this wave occlusion on our SeaDronesSee image crops before feeding them to the evidential classifier?"
* **AI Response (Summary)**: The AI suggested applying a Gaussian Blur filter (from Pillow's `ImageFilter.GaussianBlur`) with a radius dependent on the occlusion state.
* **Human Delta & Reflection**:
  * *Critical Thinking*: Gaussian blur alone doesn't capture the resolution drop of long-distance observations.
  * *Contextualization*: In real SAR, resolution degrades in proportion to the 3D distance between the camera and the target.
  * *Creative Synthesis*: I added a distance-dependent downsampling step: if the UAV is more than 25m away, the crop is downsampled to 8x8 and then upsampled back to 64x64.
  * *Decision Ownership*: Implemented this hybrid blur-downsample pipeline in `src/simulation.py`.

### Entry #: 006
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Data Understanding & Preparation
* **Problem/Context**: Transforming camera coordinate detections into sea-surface coordinates.
* **Prompt to AI**: 
  > "How do we project a target box center from image coordinates to 2D sea-surface coordinates using the camera projection matrix, and how do we compute the spatial covariance matrix propagation?"
* **AI Response (Summary)**: AI provided the pinhole camera projection equations and derived the Jacobian matrices $\mathbf{J}_p$ for pixel noise and $\mathbf{J}_{tel}$ for telemetry noise. It computed the geographic covariance $\boldsymbol{\Sigma}^{geo} = \mathbf{J}_p \boldsymbol{\Sigma}^{cam} \mathbf{J}_p^{\top} + \mathbf{J}_{tel} \boldsymbol{\Sigma}^{tel} \mathbf{J}_{tel}^{\top}$.
* **Human Delta & Reflection**:
  * *Critical Thinking*: The AI's derivation assumed a flat, static sea surface without current.
  * *Contextualization*: Real ocean currents cause targets to drift.
  * *Creative Synthesis*: I combined the projected covariance with a dynamic tracking model that incorporates current velocity.
  * *Decision Ownership*: Used this projected covariance as the measurement covariance input for our Kalman Filter tracker.

### Entry #: 007
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Data Understanding & Preparation
* **Problem/Context**: Incorporating classification uncertainty into spatial tracking.
* **Prompt to AI**: 
  > "We want to scale the Kalman Filter's measurement covariance $\mathbf{R}$ dynamically based on the classifier's epistemic uncertainty $u$. How should we write the update formula, and what is the physical meaning of the scale factor $\gamma$?"
* **AI Response (Summary)**: AI proposed the formula $\mathbf{R}_{i,t} = \boldsymbol{\Sigma}^{geo}_{i,t} + \gamma u_{i,t}\mathbf{I}_2$. It explained that $\gamma$ controls the weight of classification uncertainty on the spatial filter: higher $u$ forces the filter to rely more on the motion prediction model (drift) than on the noisy visual observation.
* **Human Delta & Reflection**:
  * *Critical Thinking*: If $\gamma$ is too high, the tracker ignores observations entirely and drifts away.
  * *Contextualization*: Over-reliance on prediction model causes state divergence.
  * *Creative Synthesis*: I determined through sweeps that $\gamma = 2.0$ is the optimal balance.
  * *Decision Ownership*: Implemented this uncertainty-scaled measurement covariance in `src/tracker.py` and `src/simulation.py`.

---

### Entry #: 008
* **Prompt Type**: PROBLEM-SOLVING
* **Stage/Component**: Exploratory Data Analysis (EDA)
* **Problem/Context**: Exploring the category distribution and size statistics in the dataset.
* **Prompt to AI**: 
  > "Can you write an EDA script to parse `instances_val.json` from SeaDronesSee and plot the distribution of bounding box areas and category labels for people in the water?"
* **AI Response (Summary)**: AI provided a matplotlib script to plot histograms of bounding box areas and bar charts of class counts.
* **Human Delta & Reflection**:
  * *Critical Thinking*: AI used generic Matplotlib styling.
  * *Contextualization*: Presentation slides and papers require matching color schemes.
  * *Creative Synthesis*: I updated the styling to use a professional, customized dark theme matching our GCS dashboard aesthetics. Wrote a wrapper script `eda_seadronessee.py` that outputs a summary report to `docs/eda_summary_report.txt`.
  * *Decision Ownership*: Used these statistics in the paper's section IV to justify the geometric feature dimensions.

### Entry #: 009
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Exploratory Data Analysis (EDA)
* **Problem/Context**: Analyzing ocean current drift vectors.
* **Prompt to AI**: 
  > "We need to simulate ocean current drift in a search area. How can we model 2D drift with constant velocity and random walk noise, and how does it affect target tracking error over 30 time steps?"
* **AI Response (Summary)**: AI suggested modeling target state transitions as $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{w}_t$, where $\mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q})$ represents random walk perturbations.
* **Human Delta & Reflection**:
  * *Critical Thinking*: Constant drift can cause targets to leave the search boundary rapidly if not compensated by the UAV path planner.
  * *Creative Synthesis*: Added boundary clipping checks to prevent targets from drifting to infinity.
  * *Decision Ownership*: I modeled the drift vector as $[0.1, -0.05]$ m/s with a small standard deviation ($0.1$m) to simulate wave agitation.

### Entry #: 010
* **Prompt Type**: PROBLEM-SOLVING
* **Stage/Component**: Exploratory Data Analysis (EDA)
* **Problem/Context**: Designing interactive visualization for the search trajectories.
* **Prompt to AI**: 
  > "We want to visualize the search grid, UAV trajectory, victim true/estimated locations, and uncertainty covariance ellipses on a Plotly map. How should we configure the shapes and traces in Plotly for a Streamlit app?"
* **AI Response (Summary)**: AI provided a Plotly Go script to add shapes (rectangles for the search area, circles for covariance boundaries, triangles for the UAV).
* **Human Delta & Reflection**:
  * *Critical Thinking*: Plotly circles generated by shapes are static and do not scale dynamically with zoom.
  * *Creative Synthesis*: I converted the covariance circles into mathematical coordinate arrays representing the 1-sigma ellipse boundary. Added color coding for targets (red for drowning, green for PFD, slate grey for filtered distractors).
  * *Decision Ownership*: Adopted this map visualization in `app.py`.

---

### Entry #: 011
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Modeling & Regression Analysis
* **Problem/Context**: Formulating the EDL model for classification.
* **Prompt to AI**: 
  > "How do we modify a PyTorch neural network to predict Dirichlet evidence parameters for classification, and what loss function (such as sum of squares or Bayes risk with KL divergence) should we use?"
* **AI Response (Summary)**: AI provided a PyTorch model with a Softplus output layer to ensure positive evidence $\mathbf{e}_k$, and implemented the EDL loss function: $L_i = L_i^{err} + \lambda_t L_i^{KL}$, where the KL term penalizes misleading evidence.
* **Human Delta & Reflection**:
  * *Critical Thinking*: The AI's KL divergence penalty could cause training instability if $\lambda_t$ is constant.
  * *Creative Synthesis*: I implemented an annealing schedule for $\lambda_t$ to slowly increase the penalty over epochs.
  * *Decision Ownership*: Used this Evidential CNN architecture in `src/models.py`.

### Entry #: 012
* **Prompt Type**: PROBLEM-SOLVING
* **Stage/Component**: Modeling & Regression Analysis
* **Problem/Context**: Training the EDL model on a lightweight feature vector instead of raw image pixels to save latency on UAV flight computers.
* **Prompt to AI**: 
  > "Since training a deep CNN on a micro-UAV is slow, we want to extract a 7-dimensional geometric feature vector from the bounding boxes to train a lightweight EDL MLP. What features should we extract, and how do we normalize them?"
* **AI Response (Summary)**: AI proposed a feature vector containing normalized width, height, aspect ratio, normalized center coordinates, distance to image center, and relative bounding box area.
* **Human Delta & Reflection**:
  * *Critical Thinking*: AI forgot that aspect ratio can become division-by-zero if height is zero.
  * *Creative Synthesis*: I added a small epsilon ($10^{-6}$) to the denominator.
  * *Decision Ownership*: Implemented this 7-dimensional extraction pipeline in `src/models.py` and trained the MLP classifier.

### Entry #: 013
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Modeling & Regression Analysis
* **Problem/Context**: Formulating the priority score for rescue routing.
* **Prompt to AI**: 
  > "How do we write a UCB-style risk-averse prioritization score that balances expected posture risk and epistemic uncertainty, while also incorporating travel distance and victim survival decay rate?"
* **AI Response (Summary)**: AI proposed the formula: $P^{rap} = \mathbb{E}[R] + \lambda u (w_{drown} - \mathbb{E}[R])$, and the final score $\operatorname{Priority} = (P^{rap} e^{\eta t_{travel}}) / (D + \epsilon)$.
* **Human Delta & Reflection**:
  * *Critical Thinking*: AI used linear distance scaling, which can cause the UAV to ignore nearby victims if their priority is slightly lower.
  * *Creative Synthesis*: I balanced the travel latency with physiological decay parameters $\eta_i$ corresponding to distress classes.
  * *Decision Ownership*: Implemented the priority scoring formula in `src/simulation.py`.

### Entry #: 014
* **Prompt Type**: PROBLEM-SOLVING
* **Stage/Component**: Modeling & Regression Analysis
* **Problem/Context**: Loading PyTorch weights into the simulation loop.
* **Prompt to AI**: 
  > "How do we load our trained PyTorch EDL CNN weights (`models/edl_weights.pth`) inside `simulation.py` to classify the cropped images from SeaDronesSee in real-time, and what is the fallback if PyTorch or the weights are missing?"
* **AI Response (Summary)**: AI provided a loader script that instantiates `EvidentialCNNClassifier(num_classes=5)`, loads the state dictionary, and falls back to a proxy simulator (`EvidentialClassifierSimulator`) if torch or the weights are unavailable.
* **Human Delta & Reflection**:
  * *Critical Thinking*: AI's script used GPU by default, which is missing on standard UAV computers.
  * *Creative Synthesis*: I forced mapping to CPU (`map_location=device`).
  * *Decision Ownership*: Integrated the PyTorch CNN loader in `src/simulation.py`.

### Entry #: 015
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Modeling & Regression Analysis
* **Problem/Context**: Designing a lookahead task scheduler for the rescue routing path optimization.
* **Prompt to AI**: 
  > "To avoid the myopic routing decisions of standard greedy priority scoring, we want to implement a lookahead trajectory scheduler of depth 3. How can we mathematically frame this path search using a joint survival probability rollout that accounts for predicted travel times and descent latencies?"
* **AI Response (Summary)**: AI suggested using a permutation-based search over the next 3 target sequences. It derived the cumulative arrival time at each node in a candidate sequence, and computed the joint VSR as the sum of exponential decay survival probabilities.
* **Human Delta & Reflection**:
  * *Critical Thinking*: The AI's permutation search was computationally heavy ($O(N!)$) and would freeze the UAV flight computer if the number of targets $N$ was large.
  * *Contextualization*: Real SAR UAVs require rapid path planning computations (under 100ms) due to limited onboard CPU and dynamic drift changes.
  * *Creative Synthesis*: I capped the search tree depth to $\min(3, \text{unrescued\_vids})$ and fell back to greedy prioritization if the number of unrescued targets exceeds 6, maintaining real-time computation bounds.
  * *Decision Ownership*: Implemented this lookahead search in `src/simulation.py` and combined it with the active descent branch trigger.

---

### Entry #: 016
* **Prompt Type**: DECISION-MAKING
* **Stage/Component**: Evaluation, Visualization & Reporting
* **Problem/Context**: Performing statistical tests to compare AES-RARR against baselines.
* **Prompt to AI**: 
  > "Reviewers want to see statistical significance tests. How do we compute the paired Wilcoxon signed-rank test and Cohen's d effect size in Python to compare the Worst VSR of AES-RARR against the distance-based router baseline over 100 trials?"
* **AI Response (Summary)**: AI provided a script using `scipy.stats.wilcoxon` to compute the p-value and a custom function to calculate Cohen's d: $d = (\mu_1 - \mu_2) / s_{pooled}$.
* **Human Delta & Reflection**:
  * *Critical Thinking*: AI's Cohen's d formula used independent sample pooling. Since the trials are paired (same starting seeds/victim configurations), this is incorrect.
  * *Creative Synthesis*: I corrected the Cohen's d formula to use the standard deviation of the paired differences.
  * *Decision Ownership*: Added this statistical testing pipeline to `run_monte_carlo.py`.

### Entry #: 017
* **Prompt Type**: PROBLEM-SOLVING
* **Stage/Component**: Evaluation, Visualization & Reporting
* **Problem/Context**: Benchmarking the pipeline components on CPU.
* **Prompt to AI**: 
  > "To prove real-time feasibility on UAVs, how do we write a script to benchmark the latency (in milliseconds) of our Kalman filter tracking step, priority computation, EDL MLP inference, and EDL CNN inference on CPU?"
* **AI Response (Summary)**: AI provided a python script using `time.perf_counter()` to run 1000 loop passes and compute the mean and standard deviation of inference times.
* **Human Delta & Reflection**:
  * *Critical Thinking*: Deep learning frameworks have a warm-up phase on first forward pass.
  * *Creative Synthesis*: I added a warm-up phase (10 passes) before starting the benchmark timer.
  * *Decision Ownership*: Implemented `run_benchmark.py` which proved that the single-pass EDL CNN takes ~5.13ms, making it suitable for onboard use.

### Entry #: 018
* **Prompt Type**: PROBLEM-SOLVING
* **Stage/Component**: Evaluation, Visualization & Reporting
* **Problem/Context**: Visualizing parameter sweeps (1D and 2D).
* **Prompt to AI**: 
  > "How can we run 1D sweeps for $\tau_{unc}$, $\lambda$, and $\gamma_R$, and a 2D sweep for $\tau_{unc} \times$ initial occlusion, and plot them as line charts and heatmaps using Seaborn?"
* **AI Response (Summary)**: AI provided a script to run simulations across parameter grids, build a pandas DataFrame, and plot them using `sns.lineplot` and `sns.heatmap`.
* **Human Delta & Reflection**:
  * *Critical Thinking*: The AI's 2D sweep script did not handle cases where the simulation failed to rescue anyone, which caused division-by-zero.
  * *Creative Synthesis*: I added robust zero-checking and clipping.
  * *Decision Ownership*: Wrote `run_sensitivity.py` and saved the output figures to `paper/figures/`.

### Entry #: 019
* **Prompt Type**: PROBLEM-SOLVING
* **Stage/Component**: Evaluation, Visualization & Reporting
* **Problem/Context**: Debugging the Streamlit duplicate ID error.
* **Prompt to AI**: 
  > "When running `app.py` in Streamlit and checking autoplay, the app crashes with `StreamlitDuplicateElementId` error on `st.plotly_chart(fig_bar, use_container_width=True)`. Why does this happen and how do I fix it?"
* **AI Response (Summary)**: AI explained that rendering Plotly charts in a loop without a unique `key` parameter causes Streamlit to generate identical element IDs, leading to a duplicate ID crash. It suggested passing `key=f"edl_bar_{vid}"` inside the loop.
* **Human Delta & Reflection**:
  * *Critical Thinking*: In addition to the key error, I found that PyArrow throws an error because the TTR column has mixed types (float and string).
  * *Creative Synthesis*: I casted TTR values to string to solve this second crash.
  * *Decision Ownership*: Modified `app.py` to fix both bugs, ensuring the GCS dashboard autoplays successfully.

---

# Hallucination Cases Detected

### Case 1: Fabricated Research Paper (Fabrication)
* **Prompt**: "Summarize 5 recent papers anomaly detection"
* **AI Response**: Listed 5 papers including 'Smith et al. (2023)' about UAV predictive maintenance.
* **Detection Method**: Searched Google Scholar and found that the paper did not exist.
* **Corrective Action**: Replaced the fabricated paper with a real paper `Jones et al. (2022)`.

### Case 2: Incorrect Theoretical Bound Claims (Logic Error)
* **Prompt**: "Explain if our Risk-UCB priority score has formal continuous regret bounds."
* **AI Response**: The AI claimed that the Risk-UCB routing formula has formal POMDP regret bounds in continuous state spaces.
* **Detection Method**: Realized that UCB in continuous, non-stationary 2D drift environments does not have formal regret guarantees.
* **Corrective Action**: Reframed the routing formula as a "myopic heuristic that approximates the value of information gain", avoiding incorrect theoretical claims.

### Case 3: Outdated/Incorrect PyTorch Optimization Call (Outdated Info)
* **Prompt**: "How can we optimize our PyTorch EDL model training loop for speed on CPU?"
* **AI Response**: Suggested using `torch.cuda.empty_cache()` inside the CPU training loop to free memory.
* **Detection Method**: Realized that `empty_cache()` is CUDA-specific and has no effect on CPU, and in fact slows down execution due to garbage collection overhead.
* **Corrective Action**: Removed the call and optimized PyTorch data loaders with `pin_memory=False` and optimized threads.
