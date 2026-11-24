# Entorno local
- Postgres aislado: `C:\Program Files\PostgreSQL\17\bin\postgres.exe -D <scratchpad>\pgdata -p 55432` (trust, user postgres, db afinia).
  Si está caído: Start-Process postgres.exe con ese -D (PowerShell, detached). NO usar pg_ctl en bash (cuelga).
- `.env` (ignorado por git) apunta ML_DATABASE_URL a esa base.
- API: `uvicorn api.main:app --port 8000` ; tests: `python -m pytest tests`.
