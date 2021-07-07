import numpy as np
import argparse
from osgeo import gdal
from pyproj import Proj


def seperate_by_date(data,resolution):
    list=[]
    date_out=[]
    for i in range(0,len(data)):
        date=str(int(data[i,0]))
        year=date[0:4]
        month=date[4:6]
        day=date[6:8]
        date_new=int(year+month+day)

        if i==0:
            list2=[]
            list2.append(data[i,1:3])
            date_old=date_new
        elif date_new<date_old+resolution:
            list2.append(data[i,1:3])
        else:
            list.append(np.array(list2))
            date_out.append(str(date_old))
            list2 = []
            list2.append(data[i, 1:3])
            date_old = date_new
    list.append(np.array(list2))
    date_out.append(str(date_old))
    return list,date_out



def lola2xy(data):

    lo=data[:,0]
    la=data[:,1]
    pa = Proj(proj='utm',zone=48,ellps='WGS84', preserve_units=False)
    # pa=Proj("EPSG:32648",preserve_units=False)
    x, y = pa(lo, la)
    out=np.array([x,y])

    return out.T

def make_map(data,width,height,min_x,max_y,resolution):
    x=data[:,0]
    y=data[:,1]
    size_x=width
    size_y=height
    map_out=np.zeros([size_x,size_y])
    for i in range(0,len(data)):
        index_x=int((x[i]-min_x)/resolution)
        index_y=int((max_y-y[i])/resolution)
        map_out[index_x,index_y]+=1
    return map_out

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input lola file')
    parser.add_argument('--output', help="output dir")
    parser.add_argument('--minlo', help="min lo")
    parser.add_argument('--maxlo', help="max lo")
    parser.add_argument('--maxla',help="max la")
    parser.add_argument('--minla',help="min la")
    args = parser.parse_args()
    input_data = np.loadtxt(args.input)
    input_data = input_data[input_data[:, 0].argsort()]
    min_lo=args.minlo
    max_lo=args.maxlo
    min_la=args.minla
    max_la=args.maxla
    temp=np.array([[min_lo,min_la],[max_lo,max_la]])
    xy=lola2xy(temp)

    min_x=int(xy[0,0])
    min_y=int(xy[0,1])
    max_x=int(xy[1,0])
    max_y=int(xy[1,1])
    resolution = 2000
    width = int(round(float((max_x - min_x)) / float(resolution)))
    height = int(round(float((max_y - min_y)) / float(resolution)))
    print ("width is %d , height is %d" %(width,height))
    ##sort accroding to the dates
    [data_list, date_list] = seperate_by_date(input_data, 1)
    for i in range(0,len(data_list)):
        name=date_list[i]+'.tif'
        print("making figure "+name)
        out = lola2xy(data_list[i])

        map_out=make_map(out,width,height,min_x,max_y,resolution)
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(args.output+'/'+name, width, height, 1, gdal.GDT_Int16)
        ds.GetRasterBand(1).WriteArray(map_out.T)
        geotransform=(min_x,resolution,0,max_y,0,-resolution)
        ds.SetGeoTransform(geotransform)
	    

