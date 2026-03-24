import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import SalaryPredictionInput

router = APIRouter()


def init_router(rf_model, mlb_languages, mlb_devtype, country_categories, feature_columns,
                language_metrics: pd.DataFrame, opportunity_index: pd.DataFrame):
    
    @router.get("/available-languages")
    def available_languages():
        """Returns all languages known by the model."""
        return sorted(mlb_languages.classes_.tolist())


    @router.get("/available-devtypes")
    def available_devtypes():
        """Returns all dev types known by the model."""
        return sorted(mlb_devtype.classes_.tolist())


    @router.get("/available-countries")
    def available_countries():
        """Returns all countries known by the model."""
        return sorted(country_categories)

    @router.post("/salary-prediction")
    def salary_prediction(data: SalaryPredictionInput):
        """Predicts salary based on languages, experience, country and dev type."""

        # Encode languages
        lang_encoded = pd.DataFrame(
            mlb_languages.transform([data.languages]),
            columns=mlb_languages.classes_,
        )

        # Encode DevType
        try:
            devtype_encoded = pd.DataFrame(
                mlb_devtype.transform([[data.dev_type]]),
                columns=mlb_devtype.classes_,
            )
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown dev_type: '{data.dev_type}'")

        # Encode Country
        if data.country not in country_categories:
            raise HTTPException(status_code=400, detail=f"Unknown country: '{data.country}'")
        country_encoded = country_categories.index(data.country)

        # Build feature row
        input_df = pd.concat(
            [
                pd.DataFrame([[data.years_of_experience, country_encoded]], columns=["YearsCode", "Country_encoded"]),
                devtype_encoded.reset_index(drop=True),
                lang_encoded.reset_index(drop=True),
            ],
            axis=1,
        )

        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        predicted_salary = float(rf_model.predict(input_df)[0])

        # Enrich response
        lang_data = []
        for lang in data.languages:
            row = language_metrics[language_metrics["Language"].str.lower() == lang.lower()]
            if not row.empty:
                entry = row.iloc[0].to_dict()
                opp = opportunity_index[opportunity_index["Language"].str.lower() == lang.lower()]
                if not opp.empty:
                    entry["growth_factor"]     = opp.iloc[0]["growth_factor"]
                    entry["opportunity_index"] = opp.iloc[0]["opportunity_index"]
                lang_data.append(entry)

        return {
            "predicted_salary": round(predicted_salary, 2),
            "input": data.model_dump(),
            "language_market_data": lang_data,
        }

    return router