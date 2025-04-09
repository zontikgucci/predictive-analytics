import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Обученная модель
model = load_model('lab1_model.h5')

test_dir = 'data/test'
class_order = ['ok_front', 'def_front']  # ok = 0, def = 1

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=1,
    class_mode='binary',
    shuffle=False,
    classes=class_order
)

predictions = model.predict(test_generator, verbose=1)
predicted_labels = (predictions > 0.5).astype(int).flatten()
true_labels = test_generator.classes
file_names = [os.path.basename(f) for f in test_generator.filenames]

# Формирование CSV-таблицы с результатами
results_df = pd.DataFrame({
    'image_name': file_names,
    'predicted_label': ['def_front' if pred == 1 else 'ok_front' for pred in predicted_labels],
    'true_label': ['def_front' if true == 1 else 'ok_front' for true in true_labels]
})

# Сохранение CSV
results_df.to_csv('lab1_results.csv', index=False)
print("Результаты предсказания сохранены в lab1_results.csv")

# Метрики
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_order))
