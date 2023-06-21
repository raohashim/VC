# Video-Coding
TU Ilmenau-Seminars Master Course Video Coding

The parameter sampler has discrete values for C (regularization) and max_iter (maximum iterations). I chose RandomParameterSampling for speed and early termination. However, if budget allows, GridParameterSampling or BayesianParameterSampling would provide more thorough exploration of the hyperparameter space.


The chosen configuration for HyperDrive is as follows:

hyperparameter_sampling: Specifies the hyperparameter sampling space.
primary_metric_name: The name of the primary metric reported by the experiment runs, which in this case is "Accuracy".
primary_metric_goal: Set to PrimaryMetricGoal.MAXIMIZE, indicating that the primary metric should be maximized during evaluation.
policy: Refers to the early termination policy that has been specified.
estimator: An estimator that will be used with the sampled hyperparameters. In this case, the estimator option was chosen, while the other two options, run_config and pipeline, were not selected. The estimator will be used in conjunction with the "train.py" file, which performs basic data manipulation.
max_total_runs: The maximum total number of runs to create. While 16 is set as the upper bound, the actual number of runs may be lower if the sample space is smaller. If both "max_total_runs" and "max_duration_minutes" are provided, the hyperparameter tuning experiment will terminate when either of these thresholds is reached.
max_concurrent_runs: Specifies the maximum number of runs to execute concurrently. If set to None, all runs are launched in parallel. The number of concurrent runs depends on the available resources in the specified compute target, so it is important to ensure that the compute target has sufficient resources for the desired concurrency.
