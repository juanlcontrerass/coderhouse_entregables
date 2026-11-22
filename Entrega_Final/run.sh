#!/usr/bin/env bash
# Levanta The Intelligence System con un solo comando.
#   ./run.sh          construye las imágenes y levanta Redis, Phoenix, la API y el worker
#   ./run.sh down     detiene todo (los datos quedan en los volúmenes)
#   ./run.sh logs     sigue los logs de la API y el worker
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-up}" in
  down) docker compose down; exit 0 ;;
  logs) docker compose logs -f api worker; exit 0 ;;
  up) ;;
  *) echo "Uso: ./run.sh [up|down|logs]"; exit 1 ;;
esac

command -v docker >/dev/null 2>&1 || { echo "Error: se necesita Docker (con Docker Compose v2)."; exit 1; }

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creó .env a partir de .env.example."
fi

key="$(grep -E '^OPENAI_API_KEY=' .env | cut -d= -f2- | tr -d '[:space:]"' || true)"
if [ -z "$key" ] || [[ "$key" == tu_* ]]; then
  echo "Error: complete OPENAI_API_KEY en el archivo .env y vuelva a ejecutar ./run.sh"
  exit 1
fi

docker compose up -d --build

echo -n "Esperando a la API"
for _ in $(seq 1 60); do
  if [ "$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q api)" 2>/dev/null)" = "healthy" ]; then
    echo
    echo "Listo."
    echo "  API y documentación:   http://localhost:8000/docs"
    echo "  Trazas (Phoenix):      http://localhost:6006"
    echo "  Demo de 5 pruebas:     python scripts/e2e_demo.py"
    echo "  Nota: el primer arranque del worker descarga el modelo de embeddings e indexa el corpus (1-2 min)."
    exit 0
  fi
  echo -n "."
  sleep 3
done
echo
echo "La API no respondió a tiempo. Revise los logs con: ./run.sh logs"
exit 1
