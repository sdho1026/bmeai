from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np  # numpy 임포트
import librosa
import h5py
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# GPU 설정 확인
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'Using GPU: {gpus[0]}')
    except RuntimeError as e:
        print(e)
else:
    print('No GPU available, using CPU instead.')

# 클래스와 라벨 설정
classes = ['bright', 'gloomy', 'calm', 'tense']
label_encoder = LabelEncoder()
label_encoder.fit(classes)

# HDF5 파일에서 데이터 로드
with h5py.File('/content/drive/MyDrive/musics/audio_data_16bit12000.h5', 'r') as hf:
    X_train = np.array(hf['X_train'])
    y_train = np.array(hf['y_train'])
    X_valid = np.array(hf['X_valid'])
    y_valid = np.array(hf['y_valid'])

# 데이터 정규화
X_train = X_train / np.max(np.abs(X_train), axis=1, keepdims=True)
X_valid = X_valid / np.max(np.abs(X_valid), axis=1, keepdims=True)

# 데이터 증강 함수
def augment_audio(audio):
    # 시간 스트레칭
    audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    # 피치 시프트
    audio = librosa.effects.pitch_shift(audio, sr=12000, n_steps=np.random.randint(-2, 2))
    # 잡음 추가
    noise = np.random.randn(len(audio))
    audio = audio + 0.005 * noise
    return audio

# 데이터 증강 적용
augmented_X_train = np.array([augment_audio(x) for x in X_train])
X_train = np.concatenate((X_train, augmented_X_train))
y_train = np.concatenate((y_train, y_train))

# 모델 정의
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.summary()

# 옵티마이저 정의 및 모델 컴파일
optimizer = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 콜백 정의
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 훈련 (배치 사이즈 16)
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_valid, y_valid),
                    callbacks=[reduce_lr, early_stopping])

# 모델 평가
test_loss, test_acc = model.evaluate(X_valid, y_valid)
print(f'Test Accuracy: {test_acc}')

# 모델을 float16으로 변환하여 저장
def save_model_as_float16(model, filepath):
    # 모델을 float16으로 변환
    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights([w.astype('float16') for w in model.get_weights()])
    # 모델 저장
    model_copy.save(filepath, include_optimizer=False)

save_model_as_float16(model, '/content/drive/MyDrive/audio_classification_model_float16.h5')