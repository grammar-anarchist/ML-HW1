### Файлы в репозитории

1. `app.py` – сервис FastAPI.
2. `utils.py` – файл к классами из ноутбука, для воспроизведения пайплайна.
3. `AI_HW1_Regression_with_inference_pro.ipynb` – ноутбук.
4. `pipeline.pkl` – сохраненный файл с пайплайном.
5. `example_json.json` и `cars_test_no_target.csv` – тестовые данные из домашки, использовались для тестирования API сервиса на видео (см. ниже); `predicted_data.csv` – результат его работы на `cars_test_no_target.csv`.

### Что было сделано?

Почти все задания, кроме бонусов. По мере усложнения модели качество практически не менялось: ни от добавления скейлера, ни от регуляризации, ни от кросс-валидации существенного прироста не было.

Единственное, что дало ощутимый прирост: добавление категориальных признаков.

По мере того, как делал домашку, упустил момент, что в конце все придется собирать в единый пайплайн ради FastAPI, поэтому пришлось перезапускать ноутбук и реализовывать все преобразования в отдельных классах. Потом выяснилось, что упустил еще более серьезный момент: при вызове uvicorn основной модуль – это __mp_main__, а не __main__, поэтому все эти классы толком нельзя использовать, т.к. они автоматически ищутся в __mp_main__, где они не определены. Если не задать их в отдельном файле, то приходится использовать костыли, чтобы они появились в __mp_main__.

В задании про FastAPI не ясно, что в итоге должно быть на входе и выходе predict_items. В задании сначала сказано, что .csv файлы, потом – что нужно дополнить код ниже, где явно используется `List[Item]` и `List[float]`, что вроде как исключает файлы. В итоге по совету из чата сделал с файлами.

### Что не сделано?

Не стал реализовывать правильное усреднение в корреляции спирмана, и не придумал свою бизнес метрику. Сначала думал взять бизнес-метрику, которая выше, и если машина выше порога, то все равно учитывать ее в результате со штрафом (чем больше завышение, тем больше штраф), а заниженные цены учитывать как 0. Потом решил, что это не очень интерпретируемо, и благополучно забыл об этом.

### Демонстрация FastAPI сервиса

Файлы example_json.json, cars_test_no_taret.csv и полученный на выходе predicted_data.csv, которые использовались для демонстрации, также присутствуют в репозитории.

[Ссылка на скринкаст с видео работы сервиса](https://disk.yandex.ru/i/Sg2Sid4RO4uo_w)
