from pydantic import BaseModel
import pandas as pd
import os
from typing import Dict, Any
import fcntl
from datetime import datetime
from fastapi import FastAPI

app = FastAPI()


# Pydantic модель для валидации входящих данных
class ResponseData(BaseModel):
    """
    Определяет Pydantic модель для валидации входящих данных. Это схема того, какие данные должен отправлять Streamlit при POST-запросе.
    participant_id: str - уникальный идентификатор участника опроса 
    scenario: str - описание сценария шока (например, "НДС +2%", "Ключевая ставка +1%")
    inflation_prediction: float - прогноз инфляции от участника в числовом виде
    timestamp: str = None - время ответа (опциональное, если не передано - добавится автоматически)
    additional_data: Dict[str, Any] = {} - дополнительные метаданные (возраст участника, регион и т.д.)
    """
    participant_id: str
    scenario: str
    inflation_prediction: float
    timestamp: str = None
    additional_data: Dict[str, Any] = {}

DATA_FILE = "data/responses.parquet"

def ensure_data_dir():
    """Создает директорию data если она не существует"""
    os.makedirs("data", exist_ok=True)

def append_to_parquet(data: dict):
    """Добавляет данные в parquet файл с file-locking"""
    ensure_data_dir()
    
    # Добавляем timestamp если не указан
    if not data.get('timestamp'):
        data['timestamp'] = datetime.now().isoformat()
    
    df_new = pd.DataFrame([data])
    
    # File locking для безопасной записи
    with open(DATA_FILE + ".lock", "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX) #ставит блокировку - только один процесс может писать; предотвращает повреждение файла при одновременных записях
        
        try:
            if os.path.exists(DATA_FILE):
                # Читаем существующие данные и добавляем новые
                df_existing = pd.read_parquet(DATA_FILE)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            
            # Сохраняем обновленные данные
            df_combined.to_parquet(DATA_FILE, index=False)
            
        finally: # finally гарантирует снятие блокировки даже при ошибке
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

@app.post("/response")
def save_response(response: ResponseData):
    """Сохраняет ответ участника опроса и передает данные в модель"""
    data = response.dict()
    append_to_parquet(data)
        
    # Подготавливаем данные для модели
    model_input = {
            "scenario": response.scenario,
            "inflation_prediction": response.inflation_prediction,
            "additional_data": response.additional_data
        }    
        
    return model_input

@app.get("/health")
def health_check():
    """Проверка работоспособности API"""
    return {"status": "healthy"}



