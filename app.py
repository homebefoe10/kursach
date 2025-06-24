import streamlit as st
import pandas as pd

# Заголовок страницы
st.set_page_config(page_title="Опрос по инфляционным ожиданиям", layout="centered")
st.title("Оценка инфляционных ожиданий")

st.markdown("""
Пожалуйста, ответьте на два вопроса:
1. **Как, на Ваш взгляд, будут меняться цены на основные потребительские товары и услуги в ближайшие один–два месяца?**  
2. **Как бы Вы оценили рост цен (инфляцию) в течение последнего месяца–двух?**
""")

# Создаём форму, чтобы при сабмите всё обрабатывалось разом
with st.form(key="inflation_survey"):
    # Вопрос 1: ожидания
    expectations = st.radio(
        "1. Как, на Ваш взгляд, будут меняться цены в ближайшие 1–2 месяца?",
        options=[
            "Сильно вырастут (>5%)",
            "Умеренно вырастут (1–5%)",
            "Останутся примерно на том же уровне",
            "Умеренно снизятся (1–5%)",
            "Сильно снизятся (>5%)"
        ]
    )

    # Вопрос 2: ретроспектива
    retrospective = st.radio(
        "2. Как бы Вы оценили рост цен за последний 1–2 месяца?",
        options=[
            "Сильно выросли (>5%)",
            "Умеренно выросли (1–5%)",
            "Остались примерно на том же уровне",
            "Умеренно снизились (1–5%)",
            "Сильно снизились (>5%)"
        ]
    )

    # Кнопка отправки
    submitted = st.form_submit_button("Отправить ответы")

if submitted:
    # 1) Подготовка данных
    responses = {
        "expectations": expectations,
        "retrospective": retrospective
    }

    # 3) (Опционально) Сохранение ответов
    df = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "expectations": expectations,
        "retrospective": retrospective,
    }])
    # Добавляем к файлу survey_results.csv
    df.to_csv("survey_results.csv", mode="a", index=False, header=not st.experimental_get_query_params().get("file_exists", False))
    # После первого сохранения можно передавать в query params флаг file_exists=True
