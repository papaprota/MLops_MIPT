Описание проекта
=====

Проект по MLOps MIPT 2024
-------

## Структура репозитория 

``` bash
├── bot.py
├── commands.py
├── configs
│   ├── __init__.py
│   ├── test.yaml
│   └── train.yaml
├── data
│   ├── gazeta_test.jsonl.dvc
│   └── gazeta_train.jsonl.dvc
├── images
│   ├── inference.png
│   └── train.png
├── logger_mlflow
│   ├── docker-compose.yaml
│   └── Dockerfile
├── README.md
├── requirements.txt
└── summarization
    ├── __init__.py
    └── scripts
        ├── datamodule.py
        ├── infer.py
        ├── __init__.py
        ├── model.py
        ├── test.py
        └── train.py
```


Формулировка задачи
------------

Проект по NLP для того, чтобы упростить жизнь пользователя, а именно меня. В последнее время из-за загружености на учебе и работе мало свободног времени. Надо оптимизировать время на чтеное новостей. Задача - суммаризация новостей в телеграм каналах (Язык пока под вопросом Русский или Английский).

Данные
-------
[Датасет с данными новостей издание Газета][1]

[Датасет с данными на русском][2]

[Датасет с новостями на русском][3]

Данных на русском языке крайне мало, 1 и 2 датасеты - это данные новостей, 2 датасет - просто немного данных для суммаризации. Только на датасете 1 уже есть обученные модели, значит идейно его хватит, но какое качество получится - вопрос хороший. Так что добавил еще 2 и 3 датасет

[Датасет новостей на английском][4]

[Датасет статей на английском с abstract][5]

[Датасет статей с Arxiv c abstract][6]

На английском данных больше, наверное, даже первого датасета хватит. Но лучше больше данных, чем меньше.


Модель
------
Так как доступные ресурсы ограничены (12 GB VRAM and 64 GB RAM), то и круг доступных моделей сужается. Все модели на Торче. 

Модели на русском

[rut5_base предобученный на данных Газеты][7]

[rut5_base][8]

Модель на английском

[t5-finetune-cnndaily-news][9]

Список моделей может поменяться (может получиться запустить модельку пожирнее)

**Обучение** - подготовить run.py file, внутри модельку обернуть в pytorch-lightning для удобного логгирования и сохранения чекпоинтов. Запустить этот .py файл в контейнере, предварительно пробросив туда volume с данными и папкой, куда сохранять модель, а так же настроить порты.

![Схема обучения](https://github.com/papaprota/MLops_MIPT/blob/main/images/train.png)

Инференс
------

Пока предполагаемая схема работы такая:

1. Юзер отправляет ссылку на канал в Telegram и даты, за которые надо суммировать информацию боту.

2. Бот отправляет текст на модель.

3. Модель отправляет сжатую информацию обратно на бота.

4. Бот отправляет суммаризированный текст юзеру в виде сообщения в Telegram.

![Схема работы](https://github.com/papaprota/MLops_MIPT/blob/main/images/inference.png)

[1]: https://github.com/IlyaGusev/gazeta
[2]: https://huggingface.co/datasets/trixdade/reviews_russian
[3]: https://huggingface.co/datasets/CarlBrendt/Summ_Dialog_News
[4]: https://huggingface.co/datasets/multi_news
[5]: https://huggingface.co/datasets/scientific_papers
[6]: https://huggingface.co/datasets/arxiv_dataset
[7]: https://huggingface.co/IlyaGusev/rut5_base_sum_gazeta/tree/main
[8]: https://huggingface.co/cointegrated/rut5-base
[9]: https://huggingface.co/minhtoan/t5-finetune-cnndaily-news/tree/main