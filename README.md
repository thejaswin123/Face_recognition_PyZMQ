# Face_recognition_PyZMQ

This application uses concept of client side and server side data transfer.</br>

The project is divided into 2 parts face recogination and client and server side data trasfer
  * Face Recognition process is defined in <a href="https://github.com/thejaswin123/Face_recognition_PyZMQ/blob/main/model.py">model.py</a>
  * For data transfer between client and server separate <a href="https://github.com/thejaswin123/Face_recognition_PyZMQ/blob/main/client.py">client.py</a> and <a href="https://github.com/thejaswin123/Face_recognition_PyZMQ/blob/main/server.py">server.py</a> is used 

# Requirements
* OpenCV
* PyZMQ

# Workflow of Program
  * At client side the input image is provided
  * The input is send to server side 
  * Server side perform the necessary recognition of person in the image.
  * Then the detected image is senf to client side and client side show the detected image
