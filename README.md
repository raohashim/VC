# Video-Coding
TU Ilmenau-Seminars Master Course Video Coding


best_hyper_run = hyperdrive_run.get_best_run_by_primary_metric()
best_hyper_run_metrics = best_hyper_run.get_metrics()
parameter_values = best_hyper_run.get_details() ['runDefinition']['arguments']

os.makedirs("./outputs", exist_ok=True)
joblib.dump(value=best_hyper_run.id,filename='outputs/best_hyper_run_model.joblib')


print('Best Run Id: ', best_hyper_run.id)
print('\n Accuracy: ', best_hyper_run_metrics['Accuracy'])
print('\n Metrics: ', best_hyper_run_metrics)
print('\n Parameters: ', parameter_values)
