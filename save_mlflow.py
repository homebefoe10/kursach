import os, mlflow
import pandas as pd
import mlflow.pyfunc
from agent import Agent 

class InflationAgentWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Ничего не грузим заранее — агент инициализируется на каждой строке
        pass

    def predict(self, api_key: str, q_num: int, model_input: pd.DataFrame) -> pd.DataFrame:
        responses = []
        for _, row in model_input.iterrows():
            agent = Agent(
                api_key=api_key,
                row=row[["PROF","SEX","AGE","FO","TIP","DOHOD","EDU"]],
                q_num=q_num
            )
            text = agent.process_query(row["query"])
            score = agent.extract_inflation_score(text)
            responses.append({"response": text, "inflation_score": score})

        return pd.DataFrame(responses)
    

def ensure_gitignore(entry: str, gitignore_path: str = ".gitignore"):
    """
    Проверяет, есть ли строка entry в .gitignore, и если нет — дописывает её.
    """
    # Если .gitignore нет, создаём его
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write(f"{entry}\n")
        return

    # Читаем существующие строки
    with open(gitignore_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    # Если нужной строки нет — дописываем
    if entry not in lines:
        with open(gitignore_path, "a") as f:
            # ставим в начало слэши, чтобы игнорировалась папка в корне
            f.write(f"\n{entry}\n")
    
    
def save_to_mlflow(artifacts: dict) -> str:
    """
    Логирует PyFunc-модель и доп. артефакты в MLflow.

    Параметры artifacts (пример):
    {
        "experiment_name": "inflation_expectations",
        "run_name": "initial_upload",
        "artifact_path": "model",
        "python_model": InflationAgentWrapper(),
        "code_paths": ["agent.py"],
        "input_example": { ... },
        "metrics_csv": "path/to/your/metrics.csv",  
        "metrics_artifact_path": "metrics"  # куда в mlruns его положить
    }
    """
    project_path = os.path.abspath(os.getcwd())
    os.environ["MLFLOW_TRACKING_URI"] = f"file:///{project_path}/mlruns"
    ensure_gitignore("mlruns/")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    mlflow.set_experiment(artifacts.get("experiment_name", "default"))

    with mlflow.start_run(run_name=artifacts.get("run_name", None)) as run:
        mlflow.pyfunc.log_model(
            artifact_path=artifacts["artifact_path"],
            python_model=artifacts["python_model"],
            code_path=artifacts.get("code_paths"),
            input_example=artifacts.get("input_example")
        )

        csv_path = artifacts.get("metrics_csv")
        if csv_path:
            # папка внутри run, куда положим CSV; по умолчанию — корень
            metrics_art_path = artifacts.get("metrics_artifact_path", "")
            mlflow.log_artifact(local_path=csv_path, artifact_path=metrics_art_path)

        run_id = run.info.run_id
        print(f"Модель и дополнительные артефакты залогированы в run: {run_id}")
        return run_id