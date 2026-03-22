from pydantic import BaseModel


class SalaryPredictionInput(BaseModel):
    languages: list[str]
    years_of_experience: float
    country: str
    dev_type: str