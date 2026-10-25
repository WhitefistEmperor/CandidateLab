# Software Engineer Interview Candidate Selection Predictor

The current user interface is HTML, CSS and JavaScript in `frontend/`. Flask in `app.py` serves it and supplies model predictions. Cleaning, training and evaluation are demonstrated from Python code and saved output files; those internal processes are no longer pages in the user interface. Start with `python app.py` and open http://127.0.0.1:8502/.

## What to tell your teacher

“This is a binary-classification demonstration for an entry-level software-engineer interview. I designed a reproducible synthetic dataset because I did not establish suitable independently collected interview data. It compares Logistic Regression, Decision Tree and Random Forest, and predicts a simulated selection outcome from candidate background and assessment scores.”

The project studies a simulation. It does not report measured hiring success or use manually collected applicant records. The generation code, assumptions and random seeds are part of the submission. All 9,000 records are programmatically generated, not individually hand-written.

## Input fields and proposed rubric

CGPA uses a 0–10 scale. Internships and projects are counts, limited to 4 and 8 in this simulation. The following assessment scores use 0–100. A zero is valid here; unlike the earlier sales dataset, we have defined the scale ourselves.

| Assessment | Proposed method for collecting a real score later |
|---|---|
| Coding correctness | Percentage of predefined test cases passed on a fixed task set |
| DSA | Rubric covering approach, correctness, complexity reasoning and edge cases, 25 points each |
| Debugging | Percentage of points earned on predefined fault-identification and repair tasks |
| OOP, DBMS, OS/networking | Percentage of points earned on fixed questions with an answer key |
| Project review | Rubric covering implementation, tests, design and ability to explain decisions, 25 points each |
| Problem solving | Rubric covering decomposition, reasoning, alternatives and verification, 25 points each |
| Communication | Rubric covering clarity, organization, explanation and response to technical questions, 25 points each |

These are proposed rubrics, not an existing validated test. The synthetic records do not contain actual submitted code or scored answers. A real collection would need fixed task difficulty, scoring anchors, independent raters and independently recorded outcomes. A score of 75 alone does not describe the difficulty of the assessment.

## Dataset generation

`generate_software_data.py` creates correlated fictional abilities, then noisy observed scores and background attributes. The invented selection process combines assessment scores with a small background contribution, an interaction and a low-coding penalty. It adds hidden noise and draws a binary label from a probability. There is no deterministic score cutoff.

This avoids a perfectly deterministic target, but the model is still learning an invented process. Adding random noise does not make synthetic data equivalent to real evidence. The exact weights, ranges and distributions are assumptions visible in the code, not facts about recruiters. We do not tune the generator to achieve a desired accuracy.

The generator exports a pristine reference containing 9,000 unique synthetic IDs. It also creates a separate raw data-entry exercise: missing values, invalid scores, label-format variants, incomplete records and duplicate copies. That copy has 9,200 rows. The errors are intentionally simulated for the cleaning demonstration, not discovered in a real organization's data.

## Cleaning, step by step

`clean_software_data.py` validates required columns, standardizes label whitespace and case, converts numeric fields and checks ranges. It rejects fractional counts for internships and projects. Invalid input values become missing so we do not invent replacement scores.

It removes exact duplicate records and quarantines conflicting IDs, unknown labels or records with more than 40% missing input fields. The 40% rule is a declared quality policy, not a learned result. It exports a cleaned CSV, a review queue, per-feature counts and a summary. The reference data is never used to recover missing scores.

Missing values remain in the cleaned CSV. This is intentional: median imputation belongs inside the training pipeline, so the test data and validation folds cannot influence the imputation values. The dataset therefore passes through both cleaning and training preprocessing.

## Training and evaluation

`software_ml.py` reserves 20% of the cleaned records for testing using a stratified split with seed 42. The remaining 80% is training data. Candidate IDs never enter the model, and duplicate candidates do not cross the split.

Each pipeline contains median imputation, standard scaling and a classifier. Five-fold cross-validation within training data selects regularization, tree depth and minimum leaf size. It fits preprocessing separately within every fold. Model selection uses CV macro F1 before test evaluation.

Logistic Regression is classification despite its name. Decision Tree learns conditions on the inputs, while Random Forest averages multiple trees. We compare these models rather than assuming that the most complex must win.

Precision and recall describe Selected predictions; ordinary F1 balances those metrics. Macro F1 averages F1 for both classes equally. Accuracy counts all correct predictions. Balanced accuracy gives both classes equal weight. The confusion matrix exposes false selections and false rejections, while the majority baseline tests whether the model improves on always choosing the most common class.

## Checking overfitting

Compare training and CV macro F1 before interpreting the test result. A large training advantage suggests memorization. Trees have limited depth and minimum leaf sizes, and Logistic Regression uses regularization. Learning curves show what happens as training size grows; they use only training data.

Close training and validation scores do not prove real-world reliability, particularly here because both sets come from the same simulator. Test metrics measure recovery of that simulation's patterns. A model can generalize within the simulation while failing on real interviews.

## Extra experiments

We compare background-only, assessment-only and combined inputs using Logistic Regression with the same split and folds. Because the generation process intentionally emphasizes assessments, any improvement reflects that design assumption. We do not claim to have discovered which traits real employers value.

Permutation importance shuffles one test feature and measures the decrease in macro F1. It describes model dependence, not causality. Local explanations use Logistic Regression's weighted inputs or Random Forest decision-path score changes. Neither turns a prediction into a validated hiring recommendation.

## Show the process in this order

1. Open the generator and metadata; explain that the data is synthetic.
2. Show raw exercise rows containing an invalid DSA score or internship count.
3. Run the cleaner and show its counts, corrected representation and review queue.
4. Open the ML pipeline and identify the split, fold-specific imputer, scaler and classifiers.
5. Show CV scores, test metrics and the confusion matrix; discuss mistakes and limitations.
6. Enter a software-engineering assessment in the app, then upload a batch.

The cleaned CSV still contains blanks; show that the pipeline fills them using training medians rather than reading the reference file. This is an important leakage-prevention choice you can explain.

## Current experiment results

Cleaning reduces 9,200 exercise rows to 8,980 candidates after removing 200 duplicate copies and quarantining 20 extensively incomplete records. It standardizes 459 label cells and leaves 2,286 missing input cells for pipeline imputation. These are measured counts after generating deliberately corrupted exercise data; corruption counts can overlap.

The split uses 7,184 training records and 1,796 test records. Logistic Regression wins CV macro F1 of 0.762. Its training macro F1 is also 0.762, and test macro F1 is 0.755, with test accuracy of 76.17%. Always predicting the majority class reaches 56.40% accuracy.

Random Forest achieves slightly higher test accuracy, 76.28%, but lower CV macro F1, 0.759, and a larger training advantage: training macro F1 is 0.807. We keep Logistic Regression because we declared training CV as the selection criterion, rather than changing the winner after inspecting the test set.

The information experiment reaches test macro F1 of 0.641 for background-only, 0.759 for assessment-only and 0.755 for combined inputs. Adding background does not improve this run. Small differences do not establish superiority across other simulations or real interview populations.

## Graphs for the demonstration

The learning curve plots training and validation macro F1 as training size increases. Explain the gap and whether extra data improves validation performance. The permutation-importance graph shows the change in test macro F1 after shuffling each input, with variation available in its tooltip. Neither graph proves real-world hiring performance.

## Reproduce everything

Run `python run_project.py` from the project folder. It generates data, cleans it, trains the models and runs checks. Then launch `python app.py`.

The tests verify repeatable generation, record accounting, conflicting-record review, training-only imputation, batch validation and forest explanations. Interface checks also verified that editing scores and resubmitting updates predictions.

## Precision-focused policy and fresh evaluation

Training out-of-fold predictions select a threshold from 0.50 to 0.90, maximizing precision subject to validation recall ≥50% and at least 50 predicted selections. These are declared design choices, not guarantees on fresh records. Hyperparameters were selected on the same training records, so development estimates can be optimistic; the fresh batch supplies the final evaluation.

We saved the model fingerprint and threshold before generating fresh labels. The threshold is 0.68 and both single and batch predictions apply it. The original classifier comparison still uses 0.50 and has its own label in the app.

On 8,980 fresh candidates from the unchanged simulator, precision increases from 70.70% to 79.80%, but recall falls from 70.61% to 49.32%. Accuracy falls from 75.51% to 73.63%. False selections decrease from 1,097 to 468; missed selections increase from 1,102 to 1,900. This is a precision–recall trade-off, not an improvement across every metric.

The 95% precision interval is approximately 78.12–81.39%. Wilson intervals describe binomial proportions; a 1,000-replicate row-count bootstrap describes F1. These intervals condition on the model and simulated population. They do not establish whether the simulator represents real hiring.

The separate noise-sensitivity experiment uses the frozen model and threshold at hidden-noise SD values of 0, 0.65 and 1.3. Selection still includes a random Bernoulli draw at SD=0. We do not change the headline dataset to pick a flattering result. If we change the policy after viewing fresh results, we must treat this batch as development data and evaluate a new untouched batch.

# Recruitment workspace update

The current entry point is `python app.py`. The home page now has HR and project manager accounts. Use `RECRUITMENT_GUIDE.md` for the role-based demonstration and `CLOUD_DEPLOYMENT.md` for AWS setup. The model-building explanation below still describes the existing ML workflow.

