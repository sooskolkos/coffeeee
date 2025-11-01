* Данный проект выполнен в качестве итогового проекта для предмета "Инжиниринг управления данными"

    * Для исследования были взяты данные из kaggle(ссылка:https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset/data).
    Также они загружены на Google Drive: https://drive.google.com/drive/folders/1r6nmAlrYXFlUtIjO5C_xPLPDvWJjBziC?hl=ru.
    
    * Для работы с данными необходимо задать окружение. Для этого использовался дистрибутив miniconda, после его установки в терминале задаётся окружение с помощью следующих команд:

    `conda create -n my_env python=3.13 pip`\
    `conda activate my_env`<br>
  
    `pip install poetry`\
    `pip install pandas`\
    `pip install seaborn`\
    `poetry new my_project`<br>
  
    `cd my_project`\
    `poetry add jupyterlab pandas matplotlib wget`<br>


    * Для написания кода использовался текстовый редактор VS Code, с созданным ранее окружением my_env. 

    * Далее был написан data_loader.py, с помощью которого подгружаются данные и etl_pipeline.py, для обработки данных.

    * Для визуализации данных был написан файл coffee.ipynb, для его создания в терминале нужно набрать команду 

    `jupiter notebook`

    * после этого будет создана ссылка на ядро с заданным окружением. 

Структура проекта: 

   my_project/<br>
├── data/<br>
│   ├── raw/<br>
│   └── processed/<br>
├── notebooks/<br>
│   └── EDA.ipynb<br>
├── src/<br>
│   ├── __init__.py<br>
│   ├── data_loader.py<br>
│   └── etl_pipeline.py<br>
├── pyproject.toml<br>
└── README.md
