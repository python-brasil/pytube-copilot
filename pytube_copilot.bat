@echo off
chcp 65001 >nul 2>&1
title PyTube Copilot v2.0.0
cd /d "%~dp0"

echo ============================================
echo        PyTube Copilot v2.0.0
echo        Iniciando aplicacao...
echo ============================================
echo.

REM Verificar se Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH.
    echo Por favor, instale o Python 3.10+ e adicione ao PATH.
    echo Download: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Verificar se o ambiente virtual existe
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Ambiente virtual nao encontrado. Criando venv...
    echo.
    python -m venv venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar o ambiente virtual.
        pause
        exit /b 1
    )
    echo [OK] Ambiente virtual criado com sucesso.
    echo.
)

REM Ativar o ambiente virtual
call venv\Scripts\activate.bat

REM Instalar/atualizar dependencias
echo [INFO] Verificando dependencias...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [AVISO] Algumas dependencias podem nao ter sido instaladas.
    echo Continuando mesmo assim...
    echo.
)

REM Instalar dependencias opcionais
pip install -q PyPDF2 openpyxl 2>nul

echo [OK] Dependencias verificadas.
echo.

REM Executar a aplicacao
echo [INFO] Iniciando PyTube Copilot...
echo.
python pytube_copilot.py

REM Desativar ambiente virtual ao fechar
deactivate 2>nul
