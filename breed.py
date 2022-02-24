#import tensorflow as tf
import cv2
import numpy as np
import tensorflow as tf

ModelPath = "./models/model.tflite"

def resolve(img):
    interpreter = tf.lite.Interpreter(model_path=ModelPath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

#    img = tf.keras.preprocessing.image.load_img(filename); #получение изображения
    img_1 = tf.keras.preprocessing.image.img_to_array(img) #конвертация изображения
    print(type(img))
    print(type(img_1))
    print(img_1.shape)
    img_1 = cv2.resize(img_1, (224, 224), interpolation=cv2.INTER_AREA) #изменение размера
    img_2 = np.expand_dims(img_1, axis=0) / 255. #перевод в матрицу

    interpreter.set_tensor(input_details[0]['index'], img_2)
    interpreter.invoke() #получение значений на нейронах выходного слоя
    output_data = interpreter.get_tensor(output_details[0]['index'])# получение результата
    y_pred_ids = output_data[0].argsort()[-5:][::-1]
    result = {}
    for i in range(len(y_pred_ids)):
        result[int(y_pred_ids[i])] = round(output_data[0][y_pred_ids[i]]*100, 5)
    return result
