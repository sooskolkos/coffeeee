#### Данный проект выполнен в качестве итогового проекта для предмета "Инжиниринг управления данными"

 Для исследования были взяты данные из kaggle(ссылка:https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset/data).
 
 Также они загружены на Google Drive: https://drive.google.com/drive/folders/1r6nmAlrYXFlUtIjO5C_xPLPDvWJjBziC?hl=ru.
    
 Для работы с данными необходимо задать окружение. Для этого использовался дистрибутив miniconda, после его установки в терминале задаётся окружение с помощью следующих команд:

    conda create -n my_env python=3.13 pip
    conda activate my_env
  
    pip install poetry
    pip install pandas
    pip install seaborn
    poetry new my_project
  
    cd my_project
    poetry add jupyterlab pandas matplotlib wget

 Для написания кода использовался текстовый редактор VS Code, с созданным ранее окружением my_env. 

 Далее был написан data_loader.py, с помощью которого подгружаются данные и etl_pipeline.py, для обработки данных.

 Для визуализации данных был написан файл coffee.ipynb, для его создания в терминале нужно набрать команду 

    jupiter notebook

 после этого будет создана ссылка на ядро с заданным окружением. 

    Структура проекта: 

    my_project/
    ├── data/
    │   ├── raw/              # Исходные данные
    │   └── processed/        # Обработанные данные
    ├── notebooks/
    │   └── EDA.ipynb        # Jupyter notebook для анализа
    ├── src/
    │   ├── __init__.py
    │   ├── data_loader.py   # Загрузка данных
    │   └── etl_pipeline.py  # ETL пайплайн
    ├── pyproject.toml       # Конфигурация Poetry
    └── README.md

