# Project 11 Causal Design

## Project

AI Causal Inference & Uplift Modeling System

## Causal Question

Among the synthetic population represented by the Project 11 dataset,
what is the estimated average effect of assignment to the training
intervention on the probability of being retained or promoted within
180 days?

## Unit of Analysis

One row represents one synthetic person.

Unique identifier:

`person_id`

The supplied dataset contains 5,000 unique synthetic persons.

The current design does not contain repeated observations for the same
person and is therefore treated as a cross-sectional observational
causal-inference problem.

## Treatment

Treatment column:

`training_assigned`

Treatment coding:

- `0` = not assigned to training
- `1` = assigned to training

The treatment is assignment to the training intervention.

It is not training completion.

## Treatment Timing

Treatment assignment occurs after measurement of the eligible
pre-treatment covariates and before post-treatment variables and the
180-day outcome.

## Outcome

Outcome column:

`retained_or_promoted_180d`

Outcome coding:

- `0` = not retained or promoted within 180 days
- `1` = retained or promoted within 180 days

The outcome is binary and occurs after treatment assignment.

## Primary Estimand

The primary estimand is the Average Treatment Effect (ATE).

ATE = E[Y(1) - Y(0)]

The target estimand represents the average difference in the probability
of being retained or promoted within 180 days under assignment to the
training intervention versus no assignment across the target synthetic
population.

The effect is an absolute probability difference.

Results should normally be reported in both probability units and
percentage points.

A positive estimate indicates a higher probability of the 180-day
outcome under treatment assignment.

A negative estimate indicates a lower probability of the 180-day
outcome under treatment assignment.

The primary estimand must not change during model development merely
because another estimand produces a more favorable result.

## Target Population

The target population is the synthetic population represented by the
Project 11 analysis dataset, subject to any later explicitly documented
overlap restrictions.

No overlap restriction is applied at this stage.

## Study Design

Project 11 is treated as a synthetic observational causal-inference
problem.

Treatment assignment may depend on observed pre-treatment covariates.

The project must therefore evaluate confounding, propensity-score
overlap, covariate balance, and adjustment quality before interpreting
an estimated effect causally.

## Eligible Pre-Treatment Covariates

The supplied variable policy identifies the following adjustment
candidates:

- `department`
- `job_level`
- `tenure_months`
- `baseline_performance`
- `baseline_engagement`
- `prior_training_hours_12m`
- `remote_work_ratio`

These variables are documented as existing before treatment assignment.

## Identifier

`person_id`

This field is used only as an identifier and must not be used as an
adjustment feature.

## Post-Treatment Variable

`post_training_completion`

This variable occurs after treatment assignment.

It must not be included as an adjustment covariate because conditioning
on a treatment-influenced variable could bias the causal estimate.

## Evaluation-Only Synthetic Ground Truth

The following fields are reserved strictly for evaluation:

- `true_propensity`
- `true_individual_treatment_effect`

These fields must not be supplied to propensity, outcome, treatment
effect, or uplift models as estimation features.

They may be used later to evaluate estimator recovery because the
dataset is synthetic.

## Identification Assumptions

A causal interpretation of the adjusted treatment-effect estimate
requires the following assumptions.

### Consistency

The defined treatment corresponds to a sufficiently well-defined
training assignment, and each observed outcome corresponds to the
potential outcome associated with the treatment actually received.

### Conditional Exchangeability

After conditioning on the eligible pre-treatment covariates, treatment
assignment and the potential outcomes are assumed to be independent.

This assumption cannot be proven from observed data.

Covariate balance does not demonstrate that unmeasured confounding is
absent.

### Positivity / Overlap

For relevant combinations of pre-treatment characteristics, there must
be a meaningful probability of observing both treated and untreated
individuals.

Propensity-score diagnostics will be used to evaluate this assumption.

No observations are trimmed or excluded at this stage.

### SUTVA / No Interference

One person's treatment assignment is assumed not to alter another
person's potential outcome.

### Correct Temporal Ordering

Adjustment covariates must exist before treatment assignment.

The treatment occurs before the defined 180-day outcome.

Post-treatment variables must not be used as confounding controls.

## Causal Interpretation Policy

The project should use language such as:

"Under the stated identification assumptions, the estimated average
treatment effect is..."

The project should not state that the training intervention definitively
caused the observed effect solely because an adjusted estimate was
produced.

## Current Status

The causal design, treatment, outcome, target population, feature
policy, temporal ordering, and primary ATE estimand have been defined.

No propensity model or adjusted causal estimator has yet been selected.

No synthetic ground-truth treatment-effect field has been used for
estimation.

No AI system has been used to calculate or modify causal estimates.