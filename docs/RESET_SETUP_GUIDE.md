# Football Predictor Model: Post-Reset Setup

This project is designed to run locally with Python and PostgreSQL. The portable project copy contains the source code, documentation, CSV/JSON data, model files that are part of the project, SQL files, and `requirements.txt`. It deliberately does not contain a Python virtual environment or installed packages.

## 1. Restore the project

Copy the portable `Football Predictor Model` folder to a local folder after Windows is reset. Do not copy a `.venv` folder from another computer. Create a new environment on the new machine instead.

## 2. Install prerequisites

Install these separately on the new computer:

- Python 3.11 or a compatible supported Python 3 version
- PostgreSQL, including the PostgreSQL command-line tools
- Git, only if repository history is needed

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency list is maintained in `requirements.txt`. Packages such as pandas, NumPy, scikit-learn, XGBoost, PuLP, Streamlit, SQLAlchemy, psycopg2-binary, matplotlib, seaborn, joblib, python-dotenv, SciPy, and requests are installed from that file rather than copied from the old computer.

## 3. Configure PostgreSQL privately

Create a local `.env` file in the project root. Never publish it or paste its values into chat:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=football_db
DB_USER=your_local_postgres_user
DB_PASS=your_local_postgres_password
```

The old computer's `.env` contains credentials and is intentionally excluded from the clean portable copy and GitHub. Use fresh or securely preserved credentials.

## 4. Restore the database backup

The separate file `postgresql_football_db.dump` is a PostgreSQL custom-format backup of the local `football_db` database. Create an empty database owned by the local PostgreSQL user, then restore it from PowerShell:

```powershell
pg_restore --clean --if-exists --no-owner --dbname=football_db postgresql_football_db.dump
```

If the database is on another host or port, provide the appropriate `--host`, `--port`, and `--username` options. Do not place passwords directly in command history; use PostgreSQL's normal private password mechanisms.

## 5. Verify before operating

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'football_predictor_pyc'
python -m py_compile app/production_dashboard.py
python -m py_compile src/production/run_weekly_pipeline.py
```

Start the read-only dashboard:

```powershell
python -m streamlit run app/production_dashboard.py --server.port 8507 --server.headless true --browser.gatherUsageStats false
```

The dashboard should handle PostgreSQL unavailability with a clear warning. Do not run the live weekly pipeline merely to test the page.

## 6. Operating rules

- Do not retrain or tune models unless that is explicitly requested as a separate task.
- Do not create fake fixtures, results, xG, predictions, or FPL history.
- Preserve Tier 1 and Tier 2 tables and applications.
- Do not add odds, sentiment, NLP, injury, manager, rivalry, style, or pressure features.
- Keep production artifacts and database credentials private.
- Use the existing pipeline and validation behavior.
