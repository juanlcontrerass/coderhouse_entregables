# Levanta The Intelligence System con un solo comando (Windows PowerShell 5.1 o PowerShell 7).
#   .\run.ps1          construye las imágenes y levanta Redis, Phoenix, la API y el worker
#   .\run.ps1 down     detiene todo (los datos quedan en los volúmenes)
#   .\run.ps1 logs     sigue los logs de la API y el worker
#   .\run.ps1 status   muestra el estado de los contenedores
# Si la política de ejecución bloquea el script, use run.cmd (mismos argumentos).

param(
    [ValidateSet('up', 'down', 'logs', 'status')]
    [string]$Command = 'up'
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Fail([string]$Message) {
    Write-Host "Error: $Message" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail 'se necesita Docker Desktop (con Docker Compose v2).'
}

switch ($Command) {
    'down'   { docker compose down; exit $LASTEXITCODE }
    'logs'   { docker compose logs -f api worker; exit $LASTEXITCODE }
    'status' { docker compose ps; exit $LASTEXITCODE }
}

# Docker Desktop instalado no implica que el motor esté corriendo
docker info --format '{{.ServerVersion}}' *> $null
if ($LASTEXITCODE -ne 0) {
    Fail 'Docker no responde. Inicie Docker Desktop y vuelva a ejecutar .\run.ps1'
}

if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Host 'Se creó .env a partir de .env.example.'
}

# Valida la clave sin mostrarla
$keyLine = Get-Content '.env' | Where-Object { $_ -match '^\s*OPENAI_API_KEY\s*=' } | Select-Object -First 1
$key = ''
if ($keyLine) { $key = ($keyLine -split '=', 2)[1].Trim().Trim('"').Trim("'") }
if (-not $key -or $key.StartsWith('tu_')) {
    Fail 'complete OPENAI_API_KEY en el archivo .env y vuelva a ejecutar .\run.ps1'
}

docker compose up -d --build
if ($LASTEXITCODE -ne 0) { Fail 'docker compose up falló. Revise el mensaje anterior.' }

Write-Host -NoNewline 'Esperando a la API'
for ($i = 0; $i -lt 60; $i++) {
    $id = docker compose ps -q api
    if ($id) {
        $health = docker inspect -f '{{.State.Health.Status}}' $id
        if ($health -eq 'healthy') {
            Write-Host ''
            Write-Host 'Listo.' -ForegroundColor Green
            Write-Host '  API y documentación:   http://localhost:8000/docs'
            Write-Host '  Trazas (Phoenix):      http://localhost:6006'
            Write-Host '  Demo de 5 pruebas:     python scripts\e2e_demo.py'
            Write-Host '  Nota: el primer arranque del worker descarga el modelo de embeddings e indexa el corpus (1-2 min).'
            exit 0
        }
    }
    Write-Host -NoNewline '.'
    Start-Sleep -Seconds 3
}
Write-Host ''
Fail 'la API no respondió a tiempo. Revise los logs con: .\run.ps1 logs'
