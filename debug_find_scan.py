import json

with open('/home/ingo/Desktop/Relational-RfDNet/datasets/splits/fullscan/scannetv2_train.json') as json_file:
    data = json.load(json_file)
    print(data[931])
