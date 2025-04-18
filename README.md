# Классификация дефектов промышленных отливок

## Задача

Цель: бинарная классификация промышленных изображений отливок на **дефектные** и **нормальные**.

Датасет используется для классификации металлической детали на **брак** и **не брак**.  
📦 **Размер**: 7300 изображений  
🔗 **Ссылка на датасет**: [Kaggle: Industrial Casting Defect Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

### Для чего это нужно:
- ✅ Снижение затрат на ручной контроль качества
- ⚡ Повышение точности и скорости обнаружения брака
- 🏭 Внедрение в реальном времени на производстве для улучшения контроля качества

## Классы

- `ok_front` — нормальные детали (**метка 0**)  
- `def_front` — детали с дефектами (**метка 1**)

---

## Архитектура модели

Используется **глубокая сверточная нейронная сеть (CNN)**.

| Слой                     | Назначение                                           |
|--------------------------|------------------------------------------------------|
| `Conv2D(32, 3x3)`        | Извлекает базовые признаки: контуры, края            |
| `MaxPooling2D(2x2)`      | Снижает размерность, оставляя важные признаки        |
| `Conv2D(64, 3x3)`        | Более сложные паттерны: текстура, формы              |
| `MaxPooling2D(2x2)`      | Снова снижает размер                                 |
| `Conv2D(128, 3x3)`       | Извлекает высокоуровневые абстрактные признаки       |
| `MaxPooling2D(2x2)`      | Сжатие признаков                                     |
| `Flatten()`              | Преобразование карты признаков в вектор              |
| `Dense(128)`             | Полносвязный слой, обучает взаимосвязи               |
| `Dropout(0.5)`           | Регуляризация, предотвращение переобучения           |
| `Dense(1, activation='sigmoid')` | Выход: вероятность дефекта (от 0 до 1)         |

### Параметры модели

- **Функции активации**: ReLU (для скрытых слоёв), Sigmoid (для выхода)  
- **Оптимизатор**: Adam  
- **Функция потерь**: Binary Crossentropy  
- **Метрика**: Accuracy

---

## Обработка и подготовка данных

### 1. Структура директорий
train/ ├── ok_front/ └── def_front/
test/ ├── ok_front/ └── def_front/
Эта структура позволяет использовать `ImageDataGenerator.flow_from_directory`, где подпапки автоматически распознаются как классы.

### 2. Предобработка

Используется `ImageDataGenerator` для:

- **Нормализации** пикселей: `rescale=1./255` (значения в диапазоне [0, 1])
- **Аугментации** (только для обучающей выборки):
  - Повороты: `rotation_range=15`
  - Зум: `zoom_range=0.1`
  - Отражение по горизонтали: `horizontal_flip=True`
- **Валидация**: 20% данных из `train` выделяются для проверки (через `validation_split=0.2`)

---

## Сохранение и использование модели

- 📁 Модель сохраняется в файл: `lab1_model.h5`
- 🔍 Может быть загружена и использована для предсказания новых изображений
- 📄 Результаты тестов сохраняются в CSV-таблицу

