FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything Streamlit needs
COPY src/streamlit_app.py  src/streamlit_app.py
COPY src/model_utils.py    src/model_utils.py
COPY models/champion_model.pkl   models/champion_model.pkl
COPY data/processed/scaler.pkl   data/processed/scaler.pkl
COPY models/results_summary.json models/results_summary.json
COPY models/evaluation.json      models/evaluation.json
COPY params.yaml .

ENV PYTHONPATH=/app

RUN adduser --disabled-password --gecos "" appuser
USER appuser



#Streamlit metrics will be on port 8501
EXPOSE 8501 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]