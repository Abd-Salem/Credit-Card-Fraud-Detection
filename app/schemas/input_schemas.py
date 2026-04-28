from pydantic import BaseModel

# transaction class
class Transaction(BaseModel):
    features : list[float]