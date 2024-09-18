# U-Net architecture for spine segmentation
def unet_model(input_size=(img_height, img_width, 1)):
    inputs = tf.keras.Input(input_size)
    
    # Contracting Path (Downsampling)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)

    # Expansive Path (Upsampling)
    u1 = layers.UpSampling2D((2, 2))(c3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D((2, 2))(c4)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    
    # Output layer for binary mask
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Compile the model
seg_model = unet_model()
seg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
