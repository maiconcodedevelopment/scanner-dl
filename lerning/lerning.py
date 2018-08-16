import numpy as np
from sqlalchemy import create_engine , String , Boolean , Enum , Integer , JSON , Table ,engine , between , modifier
import csv
import cv2

ar = csv.reader("")

input_data = np.array([2,3])

weigts = {
    'node_0' : np.array([1,1]),
    'node_1' : np.array([-1,1]),
    'node_1' : np.array([2,1])
}

nv_0 = (input_data * weigts['node_0'])
nv_1 = (input_data * weigts['node_1'])

print(sorted([34,3,34,34,3],reverse=True))

print("{0} {1}".format(nv_0,nv_1))