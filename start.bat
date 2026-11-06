@echo off
chcp 65001 >nul 2>&1
title Kiryxa — Telegram forum guard
setlocal

cd /d "%~dp0"

echo ================================
echo   Kiryxa — Telegram forum guard
echo ================================
echo.

REM ---------- 1. Python ----------
set PYTHON_BIN=
for %%P in (python python3 py) do (
    if not defined PYTHON_BIN (
        %%P --version >nul 2>&1
        if not errorlevel 1 set PYTHON_BIN=%%P
    )
)
if not defined PYTHON_BIN (
    echo [ERROR] Python не найден.
    echo Установи Python 3.10+: https://www.python.org/downloads/
    echo При установке отметь "Add Python to PATH".
    pause
    exit /b 1
)
echo [*] Python:
%PYTHON_BIN% --version

REM ---------- 2. Конфиг ----------
if not exist config.json (
    if exist config.example.json (
        copy /y config.example.json config.json >nul
        echo [!] Создан config.json — ЗАПОЛНИ bot_token, admin_tg_ids и private_chat_id.
        echo.
        notepad config.json
    )
)

if not exist config.json (
    echo [ERROR] config.json отсутствует. Скопируй config.example.json и заполни.
    pause
    exit /b 1
)

REM ---------- 3. Запуск ----------
echo [*] Запускаю бота... (Ctrl+C — остановить)
echo.
"%PYTHON_BIN%" main.py

echo.
echo [*] Бот остановлен.
pause
