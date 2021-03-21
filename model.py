import cv2
import numpy as np 
import os 

class Face_Recogination(object):

	def __init__(self):
		# self.mode=mode
		# self.test_path=test_path

		self.subjects = ["", "Thejaswin", ]

		# if self.mode == 'test':
		# 	self.video=cv2.VideoCapture(0)
		# else:
		# 	self.test_image=cv2.imread(self.test_path)

	# def __del__(self):
	# 	if self.mode=='test':
	# 		self.video.release()

	def face_detector(self,image):

		face_cascade=cv2.CascadeClassifier('opencv_files\\lbpcascade_frontalface.xml')

		# if self.mode == 'test':
			# _,image=self.video.read()
		# else:
		# print(image_path)
		# image=cv2.imread(image_path)
			# cv2.imshow("img",image)
			# cv2.waitKey(0)

		image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		faces=face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

		if (len(faces)==0):
			return None,None

		(x,y,w,h)=faces[0]
		# for (x,y,w,h) in faces:
			# cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

		# # single image is considerd here 
		roi=image[y:y+h,x:x+w]

		# cv2.imshow("roi",roi)
		# cv2.waitKey(0)
		return roi,faces[0]


	def prep_train_data(self):
		labels=[]
		faces=[]

		for label in os.listdir('train_data'):
			# labels.append(label)

			for x in os.listdir(os.path.join('train_data',label)):
				image_path=os.path.join('train_data',label,x)
				# print(image_path)
				img=cv2.imread(image_path)
				# cv2.imshow(image)
				# cv2.waitKey(0)

				face,rect=self.face_detector(img)

				if face is not None:

					if label =='Thejaswin':
						labels.append(1)

					else:
						labels.append(2)
					faces.append(face)
					# labels.append(label)
		# print(labels)
		return faces,labels
				




	def train(self):
		faces,labels=self.prep_train_data()
		# print(len(faces),len(labels))

		# arr=np.array(labels)
		# print((arr))

		self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
		self.face_recognizer.train(faces, np.array(labels))
		# self.face_recognizer.train(faces,labels)


	def predict(self,img):

		img_copy=img.copy()
		# print(self.test_path)
		face,rect=self.face_detector(img_copy)

		label=self.face_recognizer.predict(face)

		# print(label)
		label_text=self.subjects[label[0]]
		if label_text ==1:
			label_text="Thejaswin"

		(x,y,w,h)=rect

		cv2.rectangle(img_copy, (x,y), (x+w,y+h), (255,0,0),2)

		cv2.putText(img_copy, label_text, (x,y), cv2.FONT_HERSHEY_PLAIN,2.5,(0,255,0),2)
		# img_copy = cv2.resize(img_copy, (960, 540))                    # Resize image
		# cv2.imshow("img",img_copy)
		# cv2.waitKey(0)
		img_copy = cv2.resize(img_copy, (960, 540))
		print("prediction complete")
		return img_copy


# f=Face_Recogination()
# f.train()


# predicted_img=f.predict(img)
# cv2.imshow("predict",predicted_img)
# cv2.waitKey(0)
