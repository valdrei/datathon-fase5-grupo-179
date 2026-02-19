#!/bin/bash
# start.sh — Inicia FastAPI (background) + Streamlit (foreground)
# Usado no container Docker para rodar ambos os serviços no Render

# Iniciar API FastAPI em background
echo "🚀 Iniciando API FastAPI na porta 8000..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Aguardar API iniciar
sleep 5

# Iniciar Streamlit (foreground — Render monitora esse processo)
echo "🎨 Iniciando Frontend Streamlit na porta ${PORT:-8501}..."
exec python -m streamlit run frontend/app_streamlit.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.primaryColor="#667eea" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6"
