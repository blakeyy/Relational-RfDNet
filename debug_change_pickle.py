import pickle
import os

with open('datasets/scannet/processed_data/scene0126_02/bbox.pkl', 'rb') as file:
    box_info = pickle.load(file)
#with open('datasets/scannet/processed_data/scene0359_01/bbox.pkl', 'rb') as file:
#    box_info = pickle.load(file)
#with open('datasets/scannet/processed_data/scene0317_01/bbox.pkl', 'rb') as file:
#    box_info = pickle.load(file)

b = box_info[0:10]
print(b)

#for item in box_info:
#    print("Box " + str(i) + ": " + str(item['box3D']))
#    i += 1

filename = 'datasets/scannet/processed_data/scene0126_02/bbox.pkl'
outfile = open(filename,'wb')

pickle.dump(b, outfile)
outfile.close()
