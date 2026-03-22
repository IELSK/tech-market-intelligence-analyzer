# 🧠 Tech Market Intelligence Analyzer

A data platform that analyzes the programming job market and identifies career opportunities using Stack Overflow Developer Survey data.

---

## The Problem

Developers often make career decisions — which language to learn, which market to target — based on anecdotal information or outdated blog posts. This project replaces that with structured, data-driven analysis across four years of real survey data from over 270,000 developers worldwide.

---

## The Data

Data sourced from the [Stack Overflow Developer Survey](https://survey.stackoverflow.co/), covering the years 2022 through 2025. Each survey captures language usage, compensation, experience, country, and developer type from tens of thousands of respondents globally.

The raw data is not included in this repository. See [Setup](#setup) for instructions on how to download and prepare it.

---

## The Analysis

The platform answers three core questions:

**What languages are most used and best paid?**
Popularity and salary metrics are calculated across all survey years, weighted by respondent count.

**Which languages represent the best career opportunity?**
An Opportunity Index combines median salary relative to the global median, market presence, and a CAGR-based growth factor calculated from 2022 to 2025.

**How does the market differ by country?**
The same metrics are recalculated per country for a curated list of markets relevant to Brazilian developers working internationally.

---

## Architecture
```
tech-market-intelligence-analyzer/
├── config.py                     # Centralized paths and constants
├── pipeline/
│   └── ingest_stackoverflow.py   # Data ingestion and cleaning
├── analysis/
│   ├── language_analysis.py      # Global language metrics
│   ├── opportunity_index.py      # Opportunity index calculation
│   └── country_analysis.py      # Per-country metrics
├── models/
│   └── salary_predictor.py       # ML model training
├── api/
│   ├── main.py                   # FastAPI app initialization
│   ├── schemas.py                # Pydantic models
│   └── routers/                  # Endpoint modules
│       ├── languages.py
│       ├── trends.py
│       ├── countries.py
│       └── prediction.py
├── dashboard/
│   ├── app.py                    # Streamlit entry point
│   └── tabs/                     # Dashboard tab modules
│       ├── popularity.py
│       ├── opportunity.py
│       ├── trends.py
│       ├── prediction.py
│       └── countries.py
└── data/                         # Not tracked by git
    ├── raw/                      # Original survey CSVs
    ├── processed/                # Cleaned dataset (parquet)
    └── analysis/                 # Analysis outputs (parquet)
```

---

## Tech Decisions

**Pandas over Polars** — chosen for ecosystem maturity and compatibility with scikit-learn pipelines. Polars would be faster at scale but adds complexity without meaningful benefit at this data volume.

**Parquet over CSV** — preserves column types across pipeline stages and compresses significantly better. Avoids repeated type inference on every load.

**Random Forest over Gradient Boosting** — both models produced nearly identical R² scores (0.47). Random Forest was selected for its lower MAE ($31,774 vs $33,083) and faster inference time, which matters for a real-time API endpoint.

**MultiLabelBinarizer for languages and DevType** — both columns contain semicolon-separated multiple values per respondent. Binarization preserves the multi-label nature of the data and avoids the ordering artifacts of ordinal encoding.

**CAGR as growth factor** — measures compounded annual growth rate between 2022 and 2025, giving a consistent, comparable growth signal across languages regardless of their starting popularity.

**FastAPI + Streamlit** — FastAPI handles data serving and prediction with automatic schema validation and documentation. Streamlit enables rapid dashboard development without frontend overhead.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/IELSK/tech-market-intelligence-analyzer.git
cd tech-market-intelligence-analyzer
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install pandas numpy scikit-learn fastapi uvicorn streamlit plotly joblib pyarrow python-dotenv
```

### 4. Download survey data

Download the Stack Overflow Developer Survey for years 2022, 2023, 2024 and 2025 from:
👉 https://survey.stackoverflow.co/

Place each year's `survey_results_public.csv` in the corresponding folder:
```
data/raw/2022/survey_results_public.csv
data/raw/2023/survey_results_public.csv
data/raw/2024/survey_results_public.csv
data/raw/2025/survey_results_public.csv
```

### 5. Configure environment

Create a `.env` file at the project root based on `.env.example`:
```bash
cp .env.example .env
```

### 6. Run the pipeline
```bash
python pipeline/ingest_stackoverflow.py
python analysis/language_analysis.py
python analysis/opportunity_index.py
python analysis/country_analysis.py
python models/salary_predictor.py
```

---

## Running the Application

Start the API:
```bash
uvicorn api.main:app --reload
```

In a separate terminal, start the dashboard:
```bash
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`.
API documentation is available at `http://localhost:8000/docs`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/top-languages` | Languages ranked by popularity |
| GET | `/market-trends` | Languages ranked by opportunity index |
| GET | `/yearly-trends` | Language popularity per year (2022–2025) |
| GET | `/language/{name}` | Full details for a specific language |
| GET | `/country/{name}` | Top opportunities for a specific country |
| POST | `/salary-prediction` | Predict salary from developer profile |

---

## Screenshots

> Dashboard — Popularity vs Salary
> `[ screenshot placeholder ]`

> Dashboard — Opportunity Ranking
> `[ screenshot placeholder ]`

> Dashboard — Yearly Trends
> `[ screenshot placeholder ]`

> Dashboard — Salary Prediction
> `[ screenshot placeholder ]`

> Dashboard — Country Analysis
> `[ screenshot placeholder ]`

---

## Demo

> `[ demo video placeholder ]`
