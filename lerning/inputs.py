#Predição rede neural

from numpy import array , tanh , string_ , setup , matlib
import nntplib
import math
import base64
import fileinput
import csv
from sqlalchemy import *

mysql_engine = create_engine("mysql+mysqlconnector://scott:root@localhost/machine")
connection = mysql_engine.connect()

metadata = MetaData()

census = Table('cencus',metadata,
               Column('id',Integer,primary_key=True),
               )

print(repr(census))

v = array([1,4]) + array([2,5])
print(v)

weights = { 'node_0' : array([2,4]) , 'node_1' : array([4,-5]) , 'node_2' : array([2,7]) }

input_data = array([3,5])

node_0_value = ( input_data * weights['node_0'] ).sum()

node_0_output = tanh(node_0_value)

node_1_value = ( input_data * weights['node_1'] ).sum()

node_1_output = tanh(node_1_value)

hidde_layes = array([node_0_value,node_1_value])
hidde_layers_tanh = array([node_0_output,node_1_output])

output = (hidde_layes * weights['node_2']).sum(0)
output_tanh = (hidde_layers_tanh * weights['node_2']).sum(0)


print(output)
print(output_tanh)
