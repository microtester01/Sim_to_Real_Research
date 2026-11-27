Что делает каждый файл
Зависимости. 
Для подготовки KITTI и пересчёта метрик по сохранённым предсказаниям установите зависимости из requirements-kitti-tools.txt. 
Пересчёт метрик не требует PyTorch, Ultralytics или GPU.

Для запуска приведённой ниже команды KITTI-тестов дополнительно необходимы PyTorch 2.8.0, torchvision 0.23.0 и Ultralytics 8.0.200.
Выберите сборку PyTorch и torchvision, совместимую с вашими CUDA, либо установите CPU-сборку.

Команда unittest ниже запускает указанный набор KITTI-тестов, а не все тесты репозитория. 
Для полного набора тестов дополнительно требуется Python API CARLA 0.9.15, доступный выбранному интерпретатору: он нужен тестам CARLA. 
Установка CARLA API зависит от версии Python и платформы и не входит в requirements-kitti-tools.txt.

Подготовка данных и обучение:
carla_dataset.py — формирует CARLA-данные и разметку.
prepare_carla_region_split.py — создаёт пространственное разбиение CARLA.
prepare_kitti_training_data.py — подготавливает изображения и разметку KITTI по заданному manifest.
prepare_seed_initializations.py — создаёт начальные веса для случайных seed.
carla_adaptation.py — реализует обучение Baseline, DR, AFA и Combo, включая P4-ветвь и GRL.
run_experiment_batch.py — запускает три seed для выбранной конфигурации.

Оценка и разбиения:
kitti_two_class_eval.py — оценивает Pedestrian и Cyclist на KITTI и может сохранять предсказания.
carla_validation_eval.py — пересчитывает метрики CARLA-validation по сохранённым предсказаниям и GT или повторно выполняет инференс.
kitti_factorial_bootstrap.py — рассчитывает факторные эффекты, bootstrap и поправку Холма.
В каждом bootstrap-повторе одна выборка pHash-групп применяется ко всем seed-слотам и пяти конфигурациям. Равные confidence обрабатываются как один score-блок, а точечные mAP пересчитываются из сохранённых предсказаний по тому же score-block правилу, что и bootstrap-оценки.
kitti_checkpoint_manifest.py — создаёт переносимый manifest выбранных checkpoint с ID запуска, относительным путём и SHA-256.
split_manifest.json — содержит ID изображений KITTI по разбиениям.
manifest.json — описывает CARLA-разбиение.
dataset.yaml — задаёт пути и классы CARLA.
initialization_manifest.json — описывает начальные веса и их контрольные суммы.
kitti_test_phash_group_map.json — содержит соответствие тестовых изображений pHash-группам.
yolov8s_nc2_seed40.pt, yolov8s_nc2_seed41.pt, yolov8s_nc2_seed42.pt — точные начальные веса, использованные в запусках с соответствующими seed. 
Они нужны для повторения обучения с той же инициализацией. Это начальные веса моделей.

Результаты 15 запусков:
student_15run_10k_retry — содержит конфигурации, логи обучения и валидации, сведения о выбранных checkpoint, веса моделей, тестовые метрики и файлы предсказаний для пяти вариантов по трём seed.
technical_pilot.json, tests.log и carla_generator_tests.log в technical_pilot_final_20261102_v3 — отчёт технического пилота и сохранённый вывод тестов.
test_carla_adaptation.py — тестирует компоненты обучения, включая P4, GRL и потери.
test_carla_dataset.py — тестирует генерацию и преобразование CARLA-разметки.
test_kitti_two_class_eval.py — тестирует правила KITTI-оценивания.
test_kitti_factorial_bootstrap.py — тестирует bootstrap.

Из корня репозитория crossed pHash-bootstrap по сохранённым предсказаниям запускается одной командой:
python .\kitti_factorial_bootstrap.py --runs .\data\kitti_experiment\student_15run_10k_retry --manifest .\data\kitti_experiment\yolo8\split_manifest.json --labels .\data\kitti_raw\training\label_2 --predictions-dir .\data\kitti_experiment\student_15run_10k_retry\test_eval --checkpoint-manifest .\data\kitti_experiment\student_15run_10k_retry\checkpoint_manifest.json --group-map .\data\kitti_experiment\student_15run_10k_retry\test_eval\kitti_test_phash_group_map.json --output .\data\kitti_experiment\student_15run_10k_retry\test_eval\factorial_bootstrap_phash_crossed_ties_reproduced.json --draws 10000 --seed 20261108 --batch-size 32
  
Команды для повторной оценки отдельного checkpoint, например для Baseline seed 40:
python .\kitti_two_class_eval.py --checkpoint .\data\kitti_experiment\student_15run_10k_retry\Baseline_seed40\best.pt --images .\data\kitti_experiment\yolo8\images\test --labels .\data\kitti_raw\training\label_2 --split-manifest .\data\kitti_experiment\yolo8\split_manifest.json --split test --output .\data\kitti_experiment\student_15run_10k_retry\test_eval\Baseline_seed40_reproduced.json --predictions-output .\data\kitti_experiment\student_15run_10k_retry\test_eval\Baseline_seed40_reproduced_predictions.json.gz --device 0

CARLA-validation. Сохранённые предсказания 12 выбранных checkpoint, GT и ID оценочных кадров находятся в data/kitti_experiment/carla_validation_eval/reproducible_package/. 
Метрики можно пересчитать без инференса и GPU командой python .\carla_validation_eval.py --mode score. 
Для повторного инференса на локально доступных CARLA-validation изображениях и checkpoint нужно выполнить python .\carla_validation_eval.py --mode all --device 0 --batch-size 8.

Для подготовки KITTI и bootstrap требуется Python 3.10. Версии зависимостей зафиксированы в requirements-kitti-tools.txt.
requirements-kitti-tools.txt — фиксирует версии зависимостей для подготовки KITTI и bootstrap.
python -m pip install -r .\requirements-kitti-tools.txt
python .\prepare_kitti_training_data.py --split-manifest .\data\kitti_experiment\yolo8\split_manifest.json --images .\data\kitti_raw\training\image_2 --labels .\data\kitti_raw\training\label_2 --output .\data\kitti_experiment\kitti_training_subsets
python .\kitti_checkpoint_manifest.py --runs .\data\kitti_experiment\student_15run_10k_retry --output .\data\kitti_experiment\student_15run_10k_retry\checkpoint_manifest.json
python .\kitti_factorial_bootstrap.py --runs .\data\kitti_experiment\student_15run_10k_retry --manifest .\data\kitti_experiment\yolo8\split_manifest.json --labels .\data\kitti_raw\training\label_2 --predictions-dir .\data\kitti_experiment\student_15run_10k_retry\test_eval --checkpoint-manifest .\data\kitti_experiment\student_15run_10k_retry\checkpoint_manifest.json --group-map .\data\kitti_experiment\student_15run_10k_retry\test_eval\kitti_test_phash_group_map.json --output .\data\kitti_experiment\student_15run_10k_retry\test_eval\factorial_bootstrap_phash_crossed_ties_reproduced.json --draws 10000 --seed 20261108 --batch-size 32
python -m unittest test_prepare_kitti_training_data test_kitti_checkpoint_manifest test_kitti_two_class_eval test_kitti_factorial_bootstrap -v

Актуализация статистического протокола: в итоговом crossed pHash-bootstrap точечные mAP пересчитываются из сохранённых предсказаний тем же score-block правилом для равных confidence, что и bootstrap-оценки. Значения из исходных test-отчётов сохраняются в JSON только для сверки. 

