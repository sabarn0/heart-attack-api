from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type")
    trestbps: int = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=50, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)"
    )
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(
        ..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest"
    )
    slope: int = Field(..., ge=0, le=3, description="Slope of the peak exercise ST segment")
    ca: int = Field(
        ..., ge=0, le=4, description="Number of major vessels (0-4) colored by fluoroscopy"
    )
    thal: int = Field(
        ...,
        ge=0,
        le=7,
        description="Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 63,
                    "sex": 1,
                    "cp": 3,
                    "trestbps": 145,
                    "chol": 233,
                    "fbs": 1,
                    "restecg": 0,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 2.3,
                    "slope": 0,
                    "ca": 0,
                    "thal": 1,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction: 1 = disease, 0 = no disease")
    label: str = Field(..., description="Label description: 'disease' or 'no_disease'")
    confidence: float = Field(..., description="Probability of predicted class")
