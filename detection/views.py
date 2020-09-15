# Import the libraries
from django.shortcuts import render
import os
import time
import datetime
import shutil
import pandas as pd
import numpy as np
import keras
import itertools
import collections 
import pyrebase

from PIL import Image
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import OrderedDict 

from django.contrib import auth
from django.contrib.sessions.models import Session
from django.contrib.auth.models import User

import json
from json import dumps
import base64
from base64 import b64encode

import firebase_admin
from firebase import firebase
from firebase_admin import credentials
from firebase_admin import db

from .settings import BASE_DIR

# Configuring Firebase
FBConn = firebase.FirebaseApplication('https://skinai-b291a.firebaseio.com/', None)

config  = {
    'apiKey': "AIzaSyAtAXo5z4vv5iQAV83z1QrTdDo0h58lc58",
    'authDomain': "skinai-b291a.firebaseapp.com",
    'databaseURL': "https://skinai-b291a.firebaseio.com",
    'projectId': "skinai-b291a",
    'storageBucket': "skinai-b291a.appspot.com",
    'messagingSenderId': "618426768068",
    'appId': "1:618426768068:web:9ddb33ee44027d58"
  }

firebase = pyrebase.initialize_app(config)
authe = firebase.auth()

# Global variables
res2 = []
email = ""
userid = ""
image_64_encode = ""


# Main view
def index(request):
    return render(request, 'index.html')

# Sign in
def postsign(request):
	global email

	email = request.POST.get('email2')
	passw = request.POST.get('pass2')
	try:
		user = authe.sign_in_with_email_and_password(email, passw)
	except:
		message = "Invalid credentials"
		return render(request, 'index.html', {"messg": message})

		session_id = user['idToken']
		request.session['uid'] = str(session_id)
		userid = str(session_id)

	return render(request, 'testing.html')

# Log out
def logout(request):
	auth.logout(request)
	return render(request, 'index.html')

# Sign up
def signup(request):
	return render(request, 'index.html')

# Creating new user using Firebase
def postsignup(request):
	global email
	# Get the data from the template
	email=request.POST.get('email3')
	passw=request.POST.get('pass3')
	try:
		user=authe.create_user_with_email_and_password(email,passw)
	except:
		message="Unable to create account. Try again"
		return render(request,"index.html",{"messg":message})
		return render(request,"index.html")

# Skin cancer classification
def skincancer(request):
	global res2 
	global image_64_encode

	res = []

	K.clear_session()

	if request.method == 'POST':
		ts = time.time()
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H_%M_%S')

		dirPath = BASE_DIR+'/media/uploadedimg/'
		shutil.rmtree(dirPath)
		os.mkdir(dirPath)

		# Training file directory
		train_dir = os.path.join(BASE_DIR, 'train_dir')

		# Validation file directory
		val_dir = os.path.join(BASE_DIR, 'val_dir')

		# Create a MobileNet model
		mobile = keras.applications.mobilenet.MobileNet()

		# Modify the model
		# Exclude the last 5 layers of the model
		x = mobile.layers[-6].output
		# Add a dropout and dense layer for predictions
		x = Dropout(0.25)(x)
		predictions = Dense(7, activation='softmax')(x)

		# Create a new model with the new outputs
		model = Model(inputs=mobile.input, outputs=predictions)

		# Train the model
		# Define Top2 and Top3 Accuracy
		from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

		def top_3_accuracy(y_true, y_pred):
		    return top_k_categorical_accuracy(y_true, y_pred, k=3)

		def top_2_accuracy(y_true, y_pred):
		    return top_k_categorical_accuracy(y_true, y_pred, k=2)
		#Loading weights(should be in the same forlder)
		model.load_weights(BASE_DIR+'/mobile_net_final_model.h5')
		# Compile the model
		model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
		
		# Get the uploaded image	
		userImage1=request.FILES['userImage1']
		usIm = userImage1
		raw = Image.open(userImage1)
        #Temporary save the image
		imgPath1 = str(dirPath)+str(userImage1)
		raw.save(imgPath1, 'JPEG')

		# Encode the image for writing it to the DB
		with open(imgPath1, "rb") as userImage1:
			image_64_encode = base64.b64encode(userImage1.read())
		image_64_encode = str(image_64_encode, "utf-8")

		raw = np.array(raw.resize((224, 224)))/255.
		#Make sure its RGB
		raw = raw[:,:,0:3]
		#Predicting the results
		prediction = model.predict(np.expand_dims(raw, 0))
		prediction.tolist()
		Labels = ["Actinic Keratoses and intraepithelial Carcinoma","Basal Cell Carcinoma", "Benign Keratosis", "Dermatofibroma", "Melanoma","Melanocytic Nevi", "Vascular Lesions"]

		#change dictionary to list of tuples and sort
		results=dict(zip(Labels,prediction[0]))

		# dictionary sorted by value
		result_sort=OrderedDict(sorted(results.items(), key=lambda t: t[1],reverse=True))

		#Top 3 results
		result_top3 = itertools.islice(result_sort.items(), 0, 3)

		type(result_top3)

		for key in result_top3:
		    res.append(key)
		    

		for value in result_top3:
		    res.append(value)

		# Save the url of the image together with the results
		res.append(str(usIm))
		res2 = res

		K.clear_session()

	return render(request, 'testing.html', {'result1':res2})

# Saving the results to DB
def save_data(request):
	global res2 
	global email
	global image_64_encode
	global userid

	result = res2
	
	# Current date and time
	dtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Keys of the results
	key1 = result[0][0]
	key2 = result[1][0]
	key3 = result[2][0]

	#Values of the results (format them to percentage format)
	value1 = "{0:.0f}%".format(result[0][1] * 100)
	value2 = "{0:.0f}%".format(result[1][1] * 100)
	value3 = "{0:.0f}%".format(result[2][1] * 100)
    # Encoded image
	img = image_64_encode
    
    # Data that will be uploaded
	data_to_upload = {
	'datetime' : dtime,
	'user' : email,
	'key1' : key1,
	'value1' : value1,
	'key2' : key2,
	'value2' : value2,
	'key3' : key3,
	'value3' : value3,
	'image' : img
	}

	# Post the data to the appropriate folder/branch within your database
	#result = FBConn.post('/users/',email)
	#r = str(result)
	#resul = r[10:-2]
	#sr = '/users/' + resul + '/test/'
	#result4 = FBConn.post(sr,data_to_upload)
	result4 = FBConn.post('/tests/',data_to_upload)

	# Get the data from the DB
	#result2 = FBConn.get('/users', None)

	return render(request,'testing.html')
			
