import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# Dataset path
dataset_path = './split_dataset/'

# Debug directory for temporary outputs
tmp_debug_path = './tmp_debug'
print('Creating Debug Directory: ', tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)

def create_data_generators(input_size, batch_size_num):
    try:
        train_datagen = ImageDataGenerator(
            rescale=1/255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1/255)
        test_datagen = ImageDataGenerator(rescale=1/255)

        train_generator = train_datagen.flow_from_directory(
            directory=os.path.join(dataset_path, 'train'),
            target_size=(input_size, input_size),
            color_mode="rgb",
            class_mode="binary",
            batch_size=batch_size_num,
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            directory=os.path.join(dataset_path, 'val'),
            target_size=(input_size, input_size),
            color_mode="rgb",
            class_mode="binary",
            batch_size=batch_size_num,
            shuffle=True
        )

        test_generator = test_datagen.flow_from_directory(
            directory=os.path.join(dataset_path, 'test'),
            classes=['real', 'fake'],
            target_size=(input_size, input_size),
            color_mode="rgb",
            class_mode=None,
            batch_size=1,
            shuffle=False
        )

        return train_generator, val_generator, test_generator
    except Exception as e:
        print(f"Error creating data generators: {e}")
        raise

def build_model(input_size):
    try:
        efficient_net = EfficientNetB0(
            weights='imagenet',
            input_shape=(input_size, input_size, 3),
            include_top=False,
            pooling='max'
        )

        model = Sequential([
            efficient_net,
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        raise

def create_callbacks(checkpoint_filepath):
    try:
        os.makedirs(checkpoint_filepath, exist_ok=True)
        return [
            EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_filepath, 'best_model.keras'),
                monitor='val_loss',
                mode='min',
                verbose=1,
                save_best_only=True
            )
        ]
    except Exception as e:
        print(f"Error creating callbacks: {e}")
        raise

def display_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print("\nTraining Progress:")
    print("-" * 70)
    print(f"{'Epoch':<10}{'Training Accuracy':<20}{'Validation Accuracy':<20}{'Loss':<10}{'Val Loss':<10}")
    print("-" * 70)

    for epoch in range(len(acc)):
        print(f"{epoch + 1:<10}{acc[epoch]:<20.4f}{val_acc[epoch]:<20.4f}{loss[epoch]:<10.4f}{val_loss[epoch]:<10.4f}")
    print("-" * 70)

def main():
    input_size = 128
    batch_size_num = 32

    print("Creating data generators...")
    train_generator, val_generator, test_generator = create_data_generators(input_size, batch_size_num)

    print("Building the model...")
    model = build_model(input_size)

    checkpoint_filepath = './tmp_checkpoint'
    print('Creating Directory: ', checkpoint_filepath)
    os.makedirs(checkpoint_filepath, exist_ok=True)

    custom_callbacks = create_callbacks(checkpoint_filepath)

    print("Starting model training...")
    num_epochs = 20
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=custom_callbacks
    )

    print("Training complete. Displaying training history...")
    display_training_history(history)

    print("Loading the best saved model...")
    best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.keras'))
    print("Model loaded successfully.")

    print("Evaluating on test set...")
    predictions = best_model.predict(test_generator)
    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary classes
    print(f"Predictions (real=0, fake=1): {predictions}")

if __name__ == "__main__":
    main()
