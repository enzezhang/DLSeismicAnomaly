from osgeo import gdal
import numpy as np
import argparse
import os
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--filename', help='output file')
parser.add_argument('--outputfile', help='output file')
parser.add_argument('--threshold', help='the threshold for selecting the location')
parser.add_argument('--resolution', help='temporal resolution')
args = parser.parse_args()




def find_index(data,threshold):
    blur=cv2.blur(data,(3,3))*9
    index1 = np.array(np.where(data >= float(threshold)/2)).T
    index2=np.array(np.where(blur>=threshold)).T
    set_index1=set(tuple(x) for x in index1)
    set_index2=set(tuple(x) for x in index2)
    index=np.array([x for x in set_index1 & set_index2])
    return index.T


filePath=args.filename
threshold=int(args.threshold)
filename=os.path.split(filePath)[1]
temp=filename.split("_lola",1)[0]
date=temp.split("_",1)[0]
hour=int(temp.split("_",1)[1])*int(args.resolution)
dataset = gdal.Open(filePath)
adfGeoTransform = dataset.GetGeoTransform()

# print(adfGeoTransform[0])
# print(adfGeoTransform[3])

nXSize = dataset.RasterXSize
nYSize = dataset.RasterYSize
# print(nXSize)
# print(nYSize)
band=dataset.GetRasterBand(1)
data=band.ReadAsArray(0,0,nXSize,nYSize)
# index=np.array(np.where(data>=threshold))
index=find_index(data,threshold)

if len(index)==0:
    print("%s_%s cannot find the location"%(date,hour))
else:
    for i in range(index.shape[1]):

        px1 = adfGeoTransform[0] + index[1,i] * adfGeoTransform[1] +  index[0,i]* adfGeoTransform[2]
        px2=adfGeoTransform[0]  +  (index[1,i]+1) * adfGeoTransform[1] + (index[0,i])*adfGeoTransform[2]
        py1 = adfGeoTransform[3] + index[1,i] * adfGeoTransform[4] +  index[0,i] * adfGeoTransform[5]
        py2=adfGeoTransform[3] + (index[1,i])* adfGeoTransform[4] +  (index[0,i]+1) * adfGeoTransform[5]
        px=(px1+px2)/2
        py=(py1+py2)/2
        print("%s %.3f %.3f"%(date,px2,py2))
        with open(args.outputfile, 'a') as log:
            out_massage="%s %.3f %.3f %d"%(date,px,py,hour)
            log.writelines(out_massage + '\n')
