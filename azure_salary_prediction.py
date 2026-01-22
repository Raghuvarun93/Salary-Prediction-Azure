from azureml.core import Workspace, Experiment, Dataset
from azureml.core.compute import ComputeTarget
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline

# Connect to Azure ML Workspace
ws = Workspace.from_config()

# Get dataset
dataset = Dataset.get_by_name(ws, name='SalaryData')

# Get compute target
compute_target = ComputeTarget(workspace=ws, name='salary-compute')

# Create experiment
experiment = Experiment(ws, 'SalaryPredictionExperiment')

# Training step
train_step = PythonScriptStep(
    name="Train Salary Model",
    script_name="train.py",
    arguments=["--data", dataset.as_named_input('salary_data')],
    compute_target=compute_target,
    source_directory=".",
    allow_reuse=True
)

pipeline = Pipeline(workspace=ws, steps=[train_step])
run = experiment.submit(pipeline)
run.wait_for_completion(show_output=True)
