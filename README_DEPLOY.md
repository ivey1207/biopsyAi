# 🚀 Инструкция по деплою (GitHub + Hugging Face / Render)

## 1. Репозиторий на GitHub
Я уже инициализировал Git в твоей папке и сделал первый коммит. Тебе нужно только отправить код в свой новый репозиторий:

1. Создай новый **Пустой** репозиторий (Public) на [github.com/new](https://github.com/new).
2. Назови его `Biopsy_AI_Hackathon` (или как хочешь).
3. Скопируй ссылку на репозиторий (она выглядит как `https://github.com/USERNAME/REPO.git`).
4. В терминале выполни:
   ```bash
   git remote add origin ТВОЯ_ССЫЛКА_ИЗ_GITHUB
   git branch -M main
   git push -u origin main
   ```

---

## 2. Бесплатный сервис деплоймента

Я рекомендую **Hugging Face Spaces**, так как это золотой стандарт для AI хакатонов (там дают **16GB RAM** бесплатно).

### Способ А: Hugging Face Spaces (Рекомендую 🔥)
1. Зайди на [huggingface.co/new-space](https://huggingface.co/new-space).
2. Назови проект.
3. Выбери **Space SDK**: **Docker**.
4. Выбери **License**: **MIT** (или по вкусу).
5. Нажми **Create Space**.
6. Соедини его со своим GitHub репозиторием (на вкладке Settings -> Connected Repositories) или просто залей Docker-образ.
   - *Проще всего*: Выбери "Blank" Docker, а потом залей туда те же файлы, что ты залил на GitHub.
   
### Способ Б: Render.com
1. Зайди на [dashboard.render.com](https://dashboard.render.com).
2. Нажми **New** -> **Web Service**.
3. Подключи свой GitHub и выбери этот репозиторий.
4. Render сам увидит `Dockerfile`.
5. Убедись, что **Instance Type** выбран **Free**.
6. В разделе **Environment Variables** добавь:
   - `PORT`: `7860`
7. Нажми **Deploy**.

---

## 💡 Важное замечание по моделям
Твои веса моделей (`.pt`) суммарно весят около 110МБ. GitHub и Hugging Face их спокойно принимают напрямую. Если проект вырастет до 500МБ+, нужно будет использовать **Git LFS**.
