Что делает каждый файл

Подготовка данных и обучение:
carla_dataset.py — формирует CARLA-данные и разметку.
prepare_carla_region_split.py — создаёт пространственное разбиение CARLA.
prepare_kitti_training_data.py — подготавливает изображения и разметку KITTI по заданному manifest.
prepare_seed_initializations.py — создаёт начальные веса для случайных seed.
carla_adaptation.py — реализует обучение Baseline, DR, AFA и Combo, включая P4-ветвь и GRL.
run_experiment_batch.py — запускает три seed для выбранной конфигурации.

Оценка и разбиения:
kitti_two_class_eval.py — оценивает Pedestrian и Cyclist на KITTI и может сохранять предсказания.
kitti_factorial_bootstrap.py — рассчитывает факторные эффекты, bootstrap и поправку Холма.
split_manifest.json — содержит ID изображений KITTI по разбиениям.
manifest.json — описывает CARLA-разбиение.
dataset.yaml — задаёт пути и классы CARLA.
initialization_manifest.json — описывает начальные веса и их контрольные суммы.
yolov8s_nc2_seed40.pt, yolov8s_nc2_seed41.pt, yolov8s_nc2_seed42.pt — точные начальные веса, использованные в запусках с соответствующими seed. 
Они нужны для повторения обучения с той же инициализацией. Это начальные веса моделей.

Результаты 15 запусков:
student_15run_10k_retry — содержит конфигурации, логи обучения и валидации, сведения о выбранных checkpoint, веса моделей, тестовые метрики и файлы предсказаний для пяти вариантов по трём seed.
technical_pilot.json, tests.log и carla_generator_tests.log в technical_pilot_final_20261102_v3 — отчёт технического пилота и сохранённый вывод тестов.
test_carla_adaptation.py — тестирует компоненты обучения, включая P4, GRL и потери.
test_carla_dataset.py — тестирует генерацию и преобразование CARLA-разметки.
test_kitti_two_class_eval.py — тестирует правила KITTI-оценивания.
test_kitti_factorial_bootstrap.py — тестирует bootstrap.
