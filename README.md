# ENEM scores prediction using statistical models and machine learning
A two-stage modeling approach to simulate ENEM standardized test scores based on response patterns — using Item Response Theory, global optimization, and tree-like ML algorithm (XGBoost).

## What is ENEM?
ENEM (Exame Nacional do Ensino Médio) is Brazil’s national standardized test used to assess high school students' academic performance and to determine university admissions across the country. Each candidate receives four scores (Mathematics, Languages, Natural Sciences, and Human Sciences), which are not calculated from raw number of correct answers.
Instead, ENEM uses a probabilistic scoring system called Item Response Theory (IRT/TRI) to account for:
1) the difficulty of each question,
2) its ability to discriminate between students of different skill levels, and
3) the probability of guessing the right answer.

This means two candidates with the same number of correct answers can receive very different final scores depending on which questions they got right and how consistent their answer pattern is.

## Project Goal
This project implements a score simulator for ENEM 2023 using a two-stage modeling pipeline:
1) A custom optimization routine that estimates candidates’ scores using Item Response Theory — by fitting logistic functions to real response patterns.
2) A second-stage refinement using XGBoost, which incorporates uncertainty metrics from the optimization process to improve predictive accuracy.

## Dataset
We used public ENEM microdata (2019–2023) provided by INEP (Brazil’s official education data agency). Key details:

- Total size: >2GB, with over 2 million records per year.
- Initial subset: 300,000 candidates (random sample, 60k per year)
- Final modeling subset (2023 only): 100,000 candidates, balanced across all possible score values for improved generalization; response vectors standardized to a single test version (blue version)

## Modeling Pipeline
Stage 1: IRT-Based Score Estimation
For each question, we fit a 3-parameter logistic model using:
a = discrimination
b = difficulty
c = guessing probability

Parameters are estimated via global optimization using a global minimization algorithm.
Each student’s latent ability (score) is then estimated by minimizing the squared error between their response vector and the fitted IRT probabilities.
From this approximation process, we computed uncertainty metrics, such as amplitude (spread of multiple valid local minima) and second derivative of the cost function at the global minimum (indicates confidence).

Stage 2: Refinement via XGBoost
We trained a gradient boosting regressor using:
1) Estimated score from Stage 1
2) Number of correct answers
3) Uncertainty metrics (amplitude, curvature)

This refinement boosts the predictive accuracy by accounting for ambiguity in Stage 1 estimates.

## Validation Strategy
- Interwoven holdout split to maximize data efficiency: 80% training (Stage 1 and 2) + 10% validation (Stage 1 hyperparams) + 10% final test set
- Bootstrapping used for confidence intervals in Stage 1 (due to long computation time).
- 5-fold cross-validation used in Stage 2 (XGBoost) for robustness.

**Repository Structure**
.
├── dados/              # raw data from INEP
├── _banco/             # various samples used for different tasks in various data formats, depending on size
├── DICIONARIOS/        # Guides for standardizing different types of assessments.
├── METRICAS/           # Exploratory data analysis
├── MODELAGEM/          # Modelling 
├── queries e codigos de ref/   # internal team operations 
├── validacao_gabaritos/        # Documents for assessing corretude of standardization process
├── README.md


⚠️ Due to the computational cost of the optimization stage and amount of data downloaded, if you choose to download all files, we recommend using a machine with good CPU performance. 
Otherwise, all code files can be found on this repository.

[Download .zip file here] https://github.com/leticiabarbanera/practice_in_data_science_sme0852/releases/download/v1.0/Grupo.4-20250806T205116Z-1-002.zip

