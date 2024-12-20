import tensorflow as tf

# Load our model
model = tf.keras.models.load_model('animal_classifier_transfer_learning.keras')

class_labels = [
    "dog", "horse", "elephant", "butterfly", "chicken",
    "cat", "cow", "sheep", "squirrel", "spider"
]

# A helper function to preprocess the image
def preprocess_image(image_path):
    # Load the image and resize it to match the input size of the model
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img) 
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0
    return img_array

# A function to make predictions
def predicti(image_path):
    img_array = preprocess_image(image_path)

    predictions = model.predict(img_array)

    decoded_predictions = [{"label": class_labels[1], "probability": float(predictions[0][i])} for i in range(len(class_labels))]

    decoded_predictions = sorted(decoded_predictions, key=lambda x: x['probability'], reverse=True)
    
    return decoded_predictions

'''def predicti(uploads):
    """
    Realiza predicciones utilizando el modelo cargado.
    """
    img_array = preprocess_image(uploads)  # Preprocesa la imagen
    predictions = model.predict(img_array)  # Obtiene las predicciones (probabilidades)

    # Decodifica las predicciones utilizando las etiquetas de clase
    decoded_predictions = [
        {"label": class_labels[i], "probability": float(predictions[0][i])}
        for i in range(len(class_labels))
    ]

    # Ordena las predicciones por probabilidad en orden descendente
    decoded_predictions = sorted(decoded_predictions, key=lambda x: x['probability'], reverse=True)

    return decoded_predictions'''