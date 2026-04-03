# PP-OCRv5 Rec Mini Guide

Короткая инструкция для запуска обучения `PP-OCRv5` только на задаче распознавания текста (`rec`) с помощью скрипта [tools/run_ppocrv5_rec_training.py](./tools/run_ppocrv5_rec_training.py).

## 1. Установка окружения

Рекомендуется использовать `Python 3.10` и отдельное виртуальное окружение.

### Windows + GPU

```powershell
C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -m pip install -r requirements.txt
```

Проверка:

```powershell
python -c "import paddle; print('paddle=', paddle.__version__); print('cuda=', paddle.is_compiled_with_cuda()); paddle.utils.run_check()"
```

### Windows + CPU

```powershell
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
```

## 2. Подготовка алфавита

Скрипт использует строго ваш алфавит.

Есть два варианта:

1. Указать символы прямо в `ALPHABET_TEXT` в начале файла [tools/run_ppocrv5_rec_training.py](./tools/run_ppocrv5_rec_training.py)
2. Оставить `ALPHABET_TEXT = ""` и положить файл алфавита по пути `train_data/custom_ppocrv5_rec/alphabet.txt`

Поддерживаются два формата:

- один символ на строку
- или одна строка со всеми символами подряд

Пример:

```text
А
Б
В
Г
Д
...
я
0
1
2
3
4
5
6
7
8
9
-
.
,
:
(
)
 
```

Если нужен пробел, он должен присутствовать в алфавите.

## 3. Настройка скрипта

Откройте [tools/run_ppocrv5_rec_training.py](./tools/run_ppocrv5_rec_training.py) и измените только верхний блок настроек.

Основные поля:

- `DATASETS`
  список датасетов
- `MAX_TEXT_LENGTH = 40`
  максимальная длина строки
- `TRAIN_RATIO = 0.9`
  доля train, остальное уйдёт в test
- `GPUS = ""`
  для Windows лучше оставить пустым
- `RUN_TRAIN = True`
  если поставить `False`, скрипт только подготовит файлы и покажет команду
- `ALPHABET_TEXT`
  алфавит прямо в файле
- `ALPHABET_FILE`
  путь к файлу алфавита

Пример списка датасетов:

```python
DATASETS = [
    {
        "name": "CyrillicHandwritingDataset",
        "root": "CyrillicHandwritingDataset",
        "annotation": "orig_cyrillic_gt.csv",
    },
    {
        "name": "MySecondDataset",
        "root": "D:/ocr_data/MySecondDataset",
        "annotation": "labels.csv",
    },
]
```

## 4. Что скрипт делает автоматически

Скрипт:

- читает все датасеты из `DATASETS`
- собирает общий `train_list.txt`
- собирает общий `val_list.txt`
- создаёт словарь `custom_dict.txt` строго из вашего алфавита
- пропускает строки длиннее `MAX_TEXT_LENGTH`
- пропускает строки, где есть символы вне алфавита
- делает split по каждому датасету отдельно
- печатает итоговую статистику по каждому датасету

В таблице в конце выводится:

- `raw`
  сколько строк было в датасете
- `kept`
  сколько осталось после фильтрации
- `skip_len`
  сколько отброшено из-за длины
- `skip_chars`
  сколько отброшено из-за символов вне алфавита
- `train`
  сколько ушло в обучение
- `test`
  сколько ушло в проверку

## 5. Запуск

```powershell
python tools\run_ppocrv5_rec_training.py
```

## 6. Что появится после подготовки

В папке `train_data/custom_ppocrv5_rec/` будут созданы:

- `custom_dict.txt`
- `train_list.txt`
- `val_list.txt`
- `dataset_summary.json`

Чекпоинты будут сохраняться в:

```text
output/custom_ppocrv5_rec
```

## 7. Полезное замечание для Windows

На Windows лучше использовать один GPU и не включать distributed training. Поэтому в большинстве случаев:

```python
GPUS = ""
```

Если нужно сначала только проверить данные без старта обучения:

```python
RUN_TRAIN = False
```
