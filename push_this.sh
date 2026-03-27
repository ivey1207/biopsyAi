#!/bin/bash
# 🚀 Помощник для отправки кода на GitHub

# 1. Финальный коммит изменений
git add .
git commit -m "pre-deployment config: Docker, Render, README" || echo "No changes to commit"

# 2. Пуш (Нужно будет ввести логин/пароль или TOKEN от GitHub)
echo "--------------------------------------------------------"
echo "Сейчас Git запросит твой GitHub Token или пароль."
echo "Если ты не знаешь, как создать TOKEN, напиши мне: 'как создать токен?'"
echo "--------------------------------------------------------"

git push -u origin main
