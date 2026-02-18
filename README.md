<img width="1920" height="600" alt="games-popularity-analysis" src="https://github.com/user-attachments/assets/3b8d2249-38fe-4a1c-89b2-cddc952a2f49" />

***

[![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-FF6B35?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python_3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

[![NumPy](https://img.shields.io/badge/NumPy-✓-3776AB?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-✓-3776AB?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-✓-3776AB?logo=scipy&logoColor=white)](https://scipy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-✓-3776AB?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-✓-3776AB?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
***

# Анализ популярности игр интернет-магазина
По требованию заказчика проведено исследование исторических данных до 2016 года для выявления закономерностей успешности игр и формирования рекомендаций по планированию рекламных кампаний на 2017 год. Отдел маркетинга предоставил набор данных, содержащий информацию о продажах по регионам, оценках пользователей и критиков, жанрах, платформах и возрастных рейтингах ESRB

## Описание данных
Структура датафрейма `games.csv`:

- `Name` - название игры
- `Platform` - платформа (xbox, playstation, pc и др.)
- `Year_of_Release` - год выпуска
- `Genre` - жанр игры
- `NA_sales` - продажи в северной америке (в миллионах копий)
- `EU_sales` - продажи в европе (в миллионах копий)
- `JP_sales` - продажи в японии (в миллионах копий)
- `Other_sales` - продажи в других странах (в миллионах копий)
- `Critic_Score` - оценка критиков (максимум 100 баллов)
- `User_Score` - оценка пользователей (максимум 10 баллов)
- `Rating` - рейтинг ESRB (возрастная категория: e, t, m и др.)

## Результат исследования
### Исследовательский анализ
   - Самые продаваемые платформы за всё время: PS, DS, Wii, PS3, X360, PS2
   - Актуальные платформы (2014–2016): PS4, 3DS, XONE, WiiU, PS3, X360, PC, WII, PSV, PSP
   - Лидеры среди жанров: Action, Sports, Shooter, Misc, Role-Playing
   - Жизненный цикл платформ - 7 лет, новые появляются каждые 5–7 лет
###  Региональные портреты пользователей
   - **NA** - PS4 / Shooter / M
   - **EU** - PS4 / Action / M
   - **JP** - Nintendo DS / RP / не для NA
### Проверка гипотез ($\alpha < 0.05$)
   - Средние оценки пользователей PC и XONE **можно считать равными**
   - Средние оценки жанров Action и Sports **различаются**
### Рекомендации
   - Сфокусироваться на **PS4, Xbox One, Nintendo 3DS и PC** - они наиболее перспективны
   - Основные жанры для продвижения: **Action, Shooter, Sports, RP**
   - Учитывать региональные различия:
     - NA и EU - упор на PS4, M-рейтинг
     - JP - упор на Nintendo, жанр RP, игры не для NA
   - Обратить внимание на игры с рейтингом **E (для всех)**
   - Следить за жизненным циклом платформ - вовремя переключаться на новые

***

## Структура репозитория
```bash
games-popularity-analysis/
│
├── games.csv                         # Исходные данные
├── games_popularity_analysis.ipynb   # Основной ноутбук с анализом
├── games_popularity_analysis.py      # Конвертированный Python-скрипт
├── requirements.txt                  # Список зависимостей
├── README.md                         # Описание проекта
├── LICENSE                           # Лицензия
└── .gitignore                        # Исключения для Git
```

## Запуск
Клонирование репозитория
```bash
git clone https://github.com/RaffArthur/games-popularity-analysis.git

cd games-popularity-analysis
```

Установка зависимостей
```python
pip install -r requirements.txt
```

Запуск проекта
```bash
jupyter notebook games_popularity_analysis.ipynb
```
