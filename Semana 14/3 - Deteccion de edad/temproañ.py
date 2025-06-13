import tensorflow as tf

# Cargar con método legacy
model = tf.keras.models.load_model("3 - Deteccion de edad/model_gender.h5", compile=False, 
                                   custom_objects={'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D})

# Guardar en nuevo formato
model.save("model_gender_v2.h5")