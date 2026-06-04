import pandera.pandas as pa
from pydantic import BaseModel
from typing import Optional

data_schema_validator = pa.DataFrameSchema(
    {
        "Age": pa.Column(dtype=float, nullable=True),
        "Debt": pa.Column(dtype=float),
        "Married": pa.Column(dtype=int),
        "BankCustomer": pa.Column(dtype=int),
        "Industry": pa.Column(dtype=str),
        "PriorDefault": pa.Column(dtype=int),
        "YearsEmployed": pa.Column(dtype=float),
        "Employed": pa.Column(dtype=int),
        "CreditScore": pa.Column(dtype=int),
        "Citizen": pa.Column(dtype=str),
        "Income": pa.Column(dtype=int),
        "Approved": pa.Column(dtype=int),
        "ZipCode": pa.Column(dtype=str, required=False),
        "DriversLicense": pa.Column(dtype=int, required=False),
        "Gender": pa.Column(dtype=int, required=False),
        "Ethnicity": pa.Column(dtype=str, required=False),
    },
    coerce=True,
)


class DataSchema(BaseModel):
    Age: Optional[float] = None
    Debt: float
    Married: int
    BankCustomer: int
    Industry: str
    PriorDefault: int
    YearsEmployed: float
    Employed: int
    CreditScore: int
    Citizen: str
    Income: int
