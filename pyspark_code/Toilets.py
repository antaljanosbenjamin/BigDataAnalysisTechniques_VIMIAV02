# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
import pyspark.sql.types
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import pandas
import numpy as np
from pyspark.mllib.stat import Statistics 

# <codecell>

mySchema = StructType([
	StructField('ToiletID',IntegerType(),False),
	StructField('URL',StringType(),False),
	StructField('Name',StringType(),False),
	StructField('Address1',StringType(),True),
	StructField('Town',StringType(),False),
	StructField('State',StringType(),False),
	StructField('Postcode',StringType(),True),
	StructField('AddressNote',StringType(),True),
	StructField('Male',BooleanType(),False),
	StructField('Female',BooleanType(),False),
	StructField('Unisex',BooleanType(),False),
	StructField('DumpPoint',BooleanType(),False),
	StructField('FacilityType()',StringType(),True),
	StructField('ToiletType()',StringType(),True),
	StructField('AccessLimited',BooleanType(),False),
	StructField('PaymentRequired',BooleanType(),False),
	StructField('KeyRequired',BooleanType(),True),
	StructField('AccessNote',StringType(),True),
	StructField('Parking',BooleanType(),False),
	StructField('ParkingNote',StringType(),True),
	StructField('AccessibleMale',BooleanType(),False),
	StructField('AccessibleFemale',BooleanType(),False),
	StructField('AccessibleUnisex',BooleanType(),False),
	StructField('AccessibleNote',StringType(),True),
	StructField('MLAK',BooleanType(),False),
	StructField('ParkingAccessible',BooleanType(),False),
	StructField('AccessibleParkingNote',StringType(),True),
	StructField('Ambulant',BooleanType(),False),
	StructField('LHTransfer',BooleanType(),False),
	StructField('RHTransfer',BooleanType(),False),
	StructField('AdultChange',BooleanType(),False),
	StructField('IsOpen',StringType(),False),
	StructField('OpeningHoursSchedule',StringType(),True),
	StructField('OpeningHoursNote',StringType(),True),
	StructField('BabyChange',BooleanType(),False),
	StructField('Showers',BooleanType(),False),
	StructField('DrinkingWater',BooleanType(),False),
	StructField('SharpsDisposal',BooleanType(),False),
	StructField('SanitaryDisposal',StringType(),True),
	StructField('IconURL',StringType(),False),
	StructField('IconAltText',StringType(),True),
	StructField('Notes',StringType(),True),
	StructField('Status',StringType(),True),
	StructField('Latitude',DoubleType(),False),
	StructField('Longitude',DoubleType(),False)])

# <codecell>

data = spark.read.csv('input.csv', header=True, mode='DROPMALFORMED', schema=mySchema)

# <codecell>

def booleanToInteger(value):
   if value: 
        return 1
   else:
        return 0
    
def statusToInteger(value):
   if value == 'Verified': 
        return 1
   else:
        return 0

numericValues = []
data = data.withColumn('RealUnisex', (data.Male & data.Female) | data.Unisex )
data = data.withColumn('RealAccessibleUnisex', (data.AccessibleMale & data.AccessibleFemale) | data.AccessibleUnisex )
udfBooleanToInteger = udf(booleanToInteger, IntegerType())
udfStatusToInteger = udf(statusToInteger, IntegerType())
for struct in data.schema:
    if (struct.dataType == BooleanType()):
        data = data.withColumn(struct.name + 'Num', udfBooleanToInteger(struct.name))
        numericValues.append(struct.name + 'Num')
numericValues.extend(['Longitude', 'Latitude'])
data = data.withColumn('StatusNum', udfStatusToInteger('Status'))

# <codecell>

describe = data.select('RealUnisexNum', 'RealAccessibleUnisexNum').describe().drop('ToiletID')
describe.show()

# <codecell>

for col1 in numericValues:
    for col2 in numericValues:
        if not col1 < col2:
            continue
        correlation = data.corr(col1,col2)
        if abs(correlation) > 0.5:
            print '\item $corr(' + col1 + ', ' + col2 + ') = |' + str(correlation) + '|$'

# <codecell>

parkingData = data.filter('Parking')
parkingData.corr('ParkingAccessibleNum','AccessibleFemaleNum')

# <codecell>

def filterAndDrawHeatmap(datas, *args, **kwargs):
    plt.rcParams.update({'font.size' : 14})
    coords = datas.filter(kwargs.get('filterString', 'True')).select('Longitude','Latitude')
    panda = coords.toPandas()
    values = panda.get_values()
    x,y = zip(*values)
    plt.clf()
    plt.scatter(x,y,edgecolors='none',
                norm=colors.LogNorm())
    ranges = kwargs.get('ranges', [110,155,-45,-10])
    plt.axis(ranges)
    plt.title(kwargs.get('scatterTitle','Title'))
    scatterImage = kwargs.get('scatterImage',None)
    if (scatterImage is not None):
        plt.savefig(scatterImage, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    heatMax = kwargs.get('heatMax', 30)
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=kwargs.get('bins',[140,180]), range = [[ranges[0], ranges[1]], [ranges[2], ranges[3]]])
    for i in range(len(heatmap)):
            for j in range(len(heatmap[i])):
                    if heatmap[i][j] > heatMax:
                        heatmap[i][j] = heatMax
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.axis(ranges)
    plt.title(kwargs.get('heatmapTitle','Title'), fontsize=20)
    imgplot = plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.colorbar()
    heatImage = kwargs.get('heatImage',None)
    if (heatImage is not None):
        plt.savefig(heatImage, bbox_inches = 'tight', pad_inches = 0)
    return

# <codecell>

#coords = coords.select('Longitude','Latitude')
#coords = coords.filter('Longitude >= 146 or Longitude <= 144 or Latitude >= -37 or Latitude <= -39')
#coords = coords.filter('Longitude >= 152 or Longitude <= 149 or Latitude >= -32 or Latitude <= -35')
#coords = coords.filter('Longitude >= 154 or Longitude <= 151 or Latitude >= -26 or Latitude <= -29')
#.filter('Latitude <= -37').filter('Latitude >= -39')
for i in [1,2,3,4,5]:
    for j in [10,20,30,40,50]:
        filterAndDrawHeatmap(data, 
                             heatImage='heat_'+str(i)+'_'+str(j)+'.jpg',
                             heatmapTitle=str(i*35) + 'x' + str(i*45) + ' bins, maximum value is ' + str(j), 
                             heatMax=j, 
                             bins=[i*35,i*45])
        #print '\includegraphics[scale=0.35]{heat_'+str(i)+'_'+str(j) + '}'

