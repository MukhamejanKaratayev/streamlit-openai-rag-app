# Простое приложение для вопросов и ответов с использованием ИИ для документов

Интуитивно понятный инструмент на базе ИИ для извлечения и взаимодействия с текстовыми данными из PDF-документов. Используя Langchain, большие языковые модели OpenAI и Streamlit, это приложение способно читать, обрабатывать и отвечать на вопросы на основе содержимого загруженных PDF-документов.

## Содержание

- [Особенности](#features)
- [Использование](#usage)
- [Лицензия](#license)

## Особенности

- Загрузка нескольких PDF-документов.
- Извлечение и обработка текста из PDF-документов.
- Индексация и эффективный поиск текстовых данных.
- Проведение разговорного вопроса-ответа с загруженными документами.
- Настройка поведения генерации ответов ИИ.
- Очистка истории беседы по мере необходимости.

## Использование

1. Склонируйте репозиторий на локальный компьютер.
2. Перейдите в директорию проекта.
3. Установите все необходимые пакеты, выполнив команду `pip install -r requirements.txt`.
4. Создайте файл `.env` с ключом API OpenAI.
5. Запустите команду `streamlit run app.py` для запуска приложения Streamlit.
6. Откройте предоставленный URL в вашем веб-браузере.
7. Следуйте инструкциям в приложении для загрузки PDF-документов и взаимодействия с разговорным ИИ.

## Лицензия

Этот проект является предварительной версией исследования и предназначен только для некоммерческого использования, подлежит лицензии модели больших языковых моделей GPT от OpenAI, моделям встраивания HuggingFace, условиям использования данных, генерируемых OpenAI, и практике конфиденциальности Langchain.
