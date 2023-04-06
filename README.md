# CV/MLOps_ITMO_2023
## Название проекта
 Segmentation of heracleum in Kashinsky district based on satellite images Sentinel-2. 
 
 Сегментация районов распространения растения "Борщевик" с помощью алгоритмов машинного обучения на спутниковых снимках Тульской обсласти.

## Краткое описание
 Данный проект междисциплинарный, в нем приследуем несколько целей:
 - произвести сегментацию растения - борщевика в Кашинском районе (точнее в части Тульской области)
 - сравнить классический подход сегментации, в котором на вход модели подаются "условно маленькие RGB изображения" и их маски, с инструментарием библиотеки torchgeo, в которой на вход модели через загрузчик можно подавать "одно большое изображение" (спутниковый снимок) с множестовом каналов, к примеру ближний инфракрасный.
 - на реальном примере отработать техники и знания, полученные по курсам CV & MLOps
 - сделать выводы

## Этапы
Проект будет состоять из следующих этапов:
1. Скачивание и подготовка данных. Более подробно можно ознакомится в [Как создать свой датасет из космоснимков](/references/how_to_create_own_dataset.md)
2. Фиксация подхода к обучению для двух разных путей: классический и torchgeo

| Название  | Значение |
|----------:|:--------:|
| Модель    | deeplabv3_resnet50   |
| Метрика   | IoU   |
| Loss      | Dice*   |
| Аугментация данных| Нет   |
| Кол-во эпох   | 30   |
| Оптимизатор   | Lion*   |

*Можно будет попробовать провести доп.эксперимент по сравнению функций потерь **(Dice, Jaccard)**

*Можно будет попробовать провести доп.эксперимент по сравнению оптимизаторов **(Adam, AdamW, Lion)**

  
3. Проведение экспериментов. С результатами можно ознакомится [тут](/reports/README.md)
4. [Выводы](/reports/README.md)
5. Реализация MVP. Подразумевается, что итогом проекта будет web-приложение с отображением карты и слоев: предсказанного и реального. Но на первом этапе достаточно реализации через командную строку. То есть запускаем скрипт для предсказания и указываем, где находится подготовленные файлы изображения и где сохранить изображения с предсказанной маской. Более подробно с этим можно ознакомится в пункте **Порядок запуска**


# Структура папок
```
|
├── README.md                           <- Этот файл Вы читаете
├── data
│   ├── interim                         <- Папка для промежуточных данных
│   |    ├── classic                    <- Классический путь
│   |    ├── torchgeo                   <- Torchgeo путь
│   |    └── annotations.xml            <- файл разметки
│   ├── processed                       <- Папка для финальных данных
│   |    ├── classic                    <- Классический путь
│   |    └── torchgeo                   <- Torchgeo путь
│   └── raw                             <- Оригинальные данные
│
├── MLOps_course                        <- Папка для хранения выполненных ДЗ по курсу MLOps
|
├── models                              <- Папка для хранения весов моделей и др.
│   └── inference                       <- Скриншоты, картинки и другие доп.материалы
│        ├── example_imgs               <- Примеры картинок для предикта
│        └── results                    <- Папка для сохранения картинок с предсказанной маской борщевика
│
├── notebooks                           <- Jupiter - ноутбуки, возникшие в ходе проекта
│
├── references                          <- Мануалы и другие доп. материалы.
│   └── helpers                         <- Скриншоты, картинки и другие доп.материалы
│
├── reports                             <- Папка для хранения отчетов
│   ├── clearml_screens                 <- Папка для хранения скриншотов с clear-ml
│   └── PPT                             <- Папка для хранения презентаций-отчетов
│
├── poetry.lock                         <- В проекте используется poetry for python packaging and
├── poetry.toml                         <- dependency management
├── pyproject.toml
│
└── src                                 <- Папка с исходными кодами
    ├── __init__.py
    │
    ├── data                            <- Скрипты для работы с данными
    │   ├── load_from_satellite.py      <- Загрузка спутниковых снимков
    │   ├── processing_before_label.py  <- Подготовка данных для разметки
    │   ├── processing_after_label.py   <- Обработка данных после разметки
    │   ├── split_datasets.py           <- Разбиение на train & val части для 2х датасетов
    │   └── upload_data_to_clearml.py   <- Загрузка финальных данных в Clear-ML
    │
    ├── models                          <- Скрипты для моделей
    │   ├── lion_pytorch.py             <- Оптимизатор Lion для torch
    │   ├── predict_torchgeo.py         <- Предикт модели torchgeo
    │   ├── train_torchgeo.py           <- Обучение через инструменты torchgeo
    │   └── train_torch.py              <- Обучение классическим путем
    │
    │
    └── app                             <- Скрипты для web приложения
        └── app.py                      <- Приложение построенное на Fast-API
```

# Порядок запуска
1. Склонировать данный репозиторий
2. [Скачать](/models/README.md) и скопировать веса модели в папку `.\models\`
3. Скопировать подготовленный файлы (подготавливать надо самому) в папку `.\models\inference\example_imgs` или любую удобную Вам. 

**Важно**: Подготовленные картинки должны иметь 4 слой **NIR** & быть размером **256x256**

4. Запустить команду ниже
 ``` shell
python ./src/models/predict_torchgeo.py --weights ./models/test_torchgeo.pt --source ./models/inference/example_imgs --img-size 256
```
, где:
```
--weights   <путь до весов>
--source    <путь до папки с входными файлами>
--img-size  <размер подаваемых изображений>
```
5. Проверить результат в папке `./models/inference/results`

# Получившийся результат

<a href="/references/helpers/example_of_predict_torchgeo.png"><img src="/references/helpers/example_of_predict_torchgeo.png" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>

* **Красным цветом** выделены области предсказанные моделью обученая через подход **torchgeo**
* **Синим цветом** размеченная вручную область

Как видно из примера, модель справляется довольно неплохо, нашла области пропущенные при разметки, но и часть областей "потеряла" не выделив их.

Дополнительно пример предсказаний, можно посмотреть [тут](/references/helpers/output.png), но чуть в другом формате.


P.S. Данный документ будет дополняться/наполняться по мере выполнения этапов.
