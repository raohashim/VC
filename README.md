# Video-Coding
TU Ilmenau-Seminars Master Course Video Coding


automl_best_model = best_run.register_model(model_name = "best-automl-model", model_path = './outputs/model.pkl', properties={'Accuracy': best_run_metrics['accuracy']})


best_hd_model = best_hyper_run.register_model(model_name='best_hyperdrive_model', model_path='./')

