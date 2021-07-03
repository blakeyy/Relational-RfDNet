import pickle
import os

with open('datasets/scannet/processed_data/scene0126_02/bbox.pkl', 'rb') as file:
    box_info = pickle.load(file)
#with open('datasets/scannet/processed_data/scene0359_01/bbox.pkl', 'rb') as file:
#    box_info = pickle.load(file)
#with open('datasets/scannet/processed_data/scene0317_01/bbox.pkl', 'rb') as file:
#    box_info = pickle.load(file)

boxes3D = []
classes = []
shapenet_catids = []
shapenet_ids = []
object_instance_ids = []
i = 0
for item in box_info:
    print("Box " + str(i) + ": " + str(item['box3D']))
    i += 1
    