
import cv2
import socket , zmq
import numpy as np

from model import *
# We have imoprted cv2 for opening cam and showing images, zmq for communication to server, np for help in transformation



f=Face_Recogination()
f.train()



context= zmq.Context()
#Here we build the context for communication to server
socket = context.socket(zmq.REP)
#From the context we create a socket through which other clients will be connected 
socket.bind("tcp://127.0.0.1:9999")
#For the example purpose we are taking port 9999 and localhost ip for sake of demonstration
#So far we have set up connection which listens to the ip and binds with it.


# THIS FUNCTION SENDS THE FRAME IMAGE TO THE ENDPOINT
def send_array(socket, A, flags=0, copy=True, track=False):
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    # Made a dictionary of dtype of array and shape of array so that at server side we know
    # at the time of transforming array from  buffer we must know the shape to get it back.

    socket.send_json(md, flags|zmq.SNDMORE)
    #Sent the image using json to the server
    return socket.send(A, flags, copy=copy, track=track)
    # Finally sending the image in form of buffer to the server.

# THIS FUNCTION RECIEVIES THE ARRAY SENT FROM ANOTHER END
def recv_array(socket, flags=0, copy=True, track=False):
    md = socket.recv_json(flags=flags)
    # Recieves the json file containing dtype and shape of the required array
    msg = socket.recv(flags=flags, copy=copy, track=track)
    # msg is the buffer recived which contains the array
    A = np.frombuffer(msg, dtype=md["dtype"])
    # Using numpy and shape known  we transform the buffer into the array and reshape
    # it to required shape and then finally return it.
    return A.reshape(md['shape'])


image = recv_array(socket)
detected_image=f.predict(image)

    
send_array(socket,detected_image)
    # If the server is still running then we need to send the image back to the client by the socket