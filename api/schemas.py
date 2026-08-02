from pydantic import BaseModel, Field
from typing import List, Dict


class TrainRoundResponse(BaseModel):
    round: int
    global_metrics: Dict[str, float]
    per_bank_fraud_rate: Dict[str, float]


class StatusResponse(BaseModel):
    current_round: int
    history: List[TrainRoundResponse]


class PredictRequest(BaseModel):
    amount: float
    merchant_category: str
    hour: int
    day_of_week: int
    country: str
    device_type: str
    card_age_days: float
    velocity: int
    is_international: int
    is_weekend: int
    is_night: int
    is_new_card: int


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    round_used: int