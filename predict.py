import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import argparse
import json




image_path="./test_images/"



##Load the model
saved_model="my_model.h5"
model=tf.keras.models.load_model(saved_model,custom_objects={'KerasLayer':hub.KerasLayer})
model.summary()



parser = argparse.ArgumentParser()
parser.add_argument("-i","--image", help="./test_images/", required=False, default=1)
parser.add_argument("-m","--model", help="my_model.h5", required=False,default=2)
parser.add_argument("-k","--top_k", help="top k probs of the image",required=False, default=3)
parser.add_argument("-c","--category_names",help="classes",required=False,default=4)

args = vars(parser.parse_args())

image_path = args['image']
saved_model = args['model']
top_k = args['top_k']
category_names = args['category_names']
image_size = 224



# Create the process_image function
def process_image(numpy_image):
    print(numpy_image.shape)
    tensor_img=tf.image.convert_image_dtype(numpy_image, dtype=tf.int16, saturate=False)
    resized_img=tf.image.resize(numpy_image,(image_size,image_size)).numpy()
    normal_img=resized_img/255

    return normal_img    

# Create the predict function
def predict(image_path, model, top_k=0):
    #if top_k < 1:
     #   top_k = 1
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    expanded_image = np.expand_dims(image, axis=0)
    probes = model.predict(expanded_image)
    top_k_values, top_k_indices = tf.nn.top_k(probes, k=top_k)
    
    top_k_values = top_k_values.numpy()
    top_k_indices = top_k_indices.numpy()
    
    

    return top_k_values, top_k_indices, image


if category_names != None:
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    print("Classes Values:")
    top_k_values, top_k_indices = predict(image, model, top_k=int(top_k))
   # top_k_values, top_k_indices = predict(image_path, model, top_k)
    for idx in top_k_indices[0]:
        print("-",class_names[str(idx+1)])


print('Probabilties:', top_k_values)
print('Classes Keys:', top_k_indices)   
    
