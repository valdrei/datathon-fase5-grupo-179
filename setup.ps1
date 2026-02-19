# Script de Setup do Projeto Passos Mágicos
# Execute: .\setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Passos Mágicos - Setup do Projeto" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Python
Write-Host "1. Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ Python encontrado: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "   ✗ Python não encontrado! Instale Python 3.11+" -ForegroundColor Red
    exit 1
}

# Criar ambiente virtual
Write-Host ""
Write-Host "2. Criando ambiente virtual..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "   ⚠ Ambiente virtual já existe" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "   ✓ Ambiente virtual criado" -ForegroundColor Green
}

# Ativar ambiente virtual
Write-Host ""
Write-Host "3. Ativando ambiente virtual..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "   ✓ Ambiente ativado" -ForegroundColor Green

# Instalar dependências
Write-Host ""
Write-Host "4. Instalando dependências..." -ForegroundColor Yellow
pip install --upgrade pip -q
pip install -r requirements.txt -q
Write-Host "   ✓ Dependências instaladas" -ForegroundColor Green

# Verificar arquivos de dados
Write-Host ""
Write-Host "5. Verificando arquivos de dados..." -ForegroundColor Yellow
$dataFiles = @("data\PEDE2022.csv", "data\PEDE2023.csv", "data\PEDE2024.csv")
$allFound = $true
foreach ($dataFile in $dataFiles) {
    if (Test-Path $dataFile) {
        Write-Host "   ✓ $dataFile encontrado" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ $dataFile não encontrado" -ForegroundColor Yellow
        $allFound = $false
    }
}
if (-not $allFound) {
    Write-Host "     Execute: python exportar_excel.py" -ForegroundColor Yellow
}

# Criar diretórios necessários
Write-Host ""
Write-Host "6. Verificando estrutura de diretórios..." -ForegroundColor Yellow
$dirs = @("app\model", "logs", "data", "notebooks")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}
Write-Host "   ✓ Diretórios verificados" -ForegroundColor Green

# Resumo
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Concluído!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Próximos passos:" -ForegroundColor White
Write-Host "  1. Exportar dados do Excel (se necessário):" -ForegroundColor White
Write-Host "     python exportar_excel.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Treinar o modelo:" -ForegroundColor White
Write-Host "     python -m src.train" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Iniciar a API:" -ForegroundColor White
Write-Host "     python -m uvicorn app.main:app --reload" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Rodar testes:" -ForegroundColor White
Write-Host "     pytest tests/ -v" -ForegroundColor Gray
Write-Host ""
Write-Host "Para mais informações, veja README.md" -ForegroundColor Cyan
Write-Host ""
