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
	StructField('FacilityType',StringType(),True),
	StructField('ToiletType',StringType(),True),
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
data = data.withColumn('OnlyMale', (data.Male & ~( data.Female | data.Unisex ) ) )
data = data.withColumn('OnlyFemale', (data.Female & ~( data.Male | data.Unisex ) ) )
data = data.withColumn('UnisexAndMale', (data.Male & data.Unisex & ~ data.Female ) )
data = data.withColumn('UnisexAndFemale', (data.Female & data.Unisex & ~ data.Male ) )
data = data.withColumn('AccessibleOnlyMale', (data.AccessibleMale & ~( data.AccessibleFemale | data.AccessibleUnisex ) ) )
data = data.withColumn('AccessibleOnlyFemale', (data.AccessibleFemale & ~( data.AccessibleMale | data.AccessibleUnisex ) ) )
data = data.withColumn('AccessibleUnisexAndMale', (data.AccessibleMale & data.AccessibleUnisex & ~ data.AccessibleFemale ) )
data = data.withColumn('AccessibleUnisexAndFemale', (data.AccessibleFemale & data.AccessibleUnisex & ~ data.AccessibleMale ) )
data = data.withColumn('OnlyAccessible',  ( ~(data.Male | data.Female | data.Unisex ) & (data.AccessibleFemale | data.AccessibleUnisex | data.AccessibleMale )) )
data = data.withColumn('AtLeastOneAccessible', (data.AccessibleFemale | data.AccessibleUnisex | data.AccessibleMale ) )
udfBooleanToInteger = udf(booleanToInteger, IntegerType())
udfStatusToInteger = udf(statusToInteger, IntegerType())
for struct in data.schema:
    if (struct.dataType == BooleanType()):
        data = data.withColumn(struct.name + 'Num', udfBooleanToInteger(struct.name))
        numericValues.append(struct.name + 'Num')
numericValues.extend(['Longitude', 'Latitude'])
data = data.withColumn('StatusNum', udfStatusToInteger('Status'))

# <codecell>

for col1 in numericValues:
    for col2 in numericValues:
        if not col1 < col2:
            continue
        correlation = data.corr(col1,col2)
        if abs(correlation) > 0.5:
            print '\item $corr(' + col1 + ', ' + col2 + ') = |' + str(correlation) + '|$'

# <codecell>

def filterAndDrawScatter(datas, *args, **kwargs):
    plt.rcParams.update({'font.size' : 14})
    coords = datas.filter(kwargs.get('filterString', 'True')).select('Longitude','Latitude')
    panda = coords.toPandas()
    values = panda.get_values()
    x,y = zip(*values)
    plt.clf()
    alpha = kwargs.get('alpha',1.0)
    marker = kwargs.get('marker', 'o')
    ranges = kwargs.get('ranges', [110,155,-45,-10])
    title = kwargs.get('scatterTitle',' ')
    color = kwargs.get('color', 'blue')
    
    plt.scatter(x,y, edgecolors="none", alpha = alpha, color = color)
    plt.axis(ranges)
    plt.title(title, fontsize = 18)
    scatterImage = kwargs.get('scatterImage',None)
    if (scatterImage is not None):
        plt.savefig(scatterImage, bbox_inches = 'tight', pad_inches = 0)
    if kwargs.get('showScatter',True):
        plt.show()
    return

def filterAndDrawHeatmap(datas, *args, **kwargs):
    plt.rcParams.update({'font.size' : 14})
    coords = datas.filter(kwargs.get('filterString', 'True')).select('Longitude','Latitude')
    panda = coords.toPandas()
    values = panda.get_values()
    x,y = zip(*values)
    heatMax = kwargs.get('heatMax', 30)
    ranges = kwargs.get('ranges', [110,155,-45,-10])
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=kwargs.get('bins',[140/4*3,180/4*3]), range = [[ranges[0], ranges[1]], [ranges[2], ranges[3]]])
    for i in range(len(heatmap)):
            for j in range(len(heatmap[i])):
                    if heatmap[i][j] > heatMax:
                        heatmap[i][j] = heatMax
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.axis(ranges)
    plt.title(kwargs.get('heatmapTitle',' '), fontsize=18)
    imgplot = plt.imshow(heatmap.T, extent=extent, origin='lower', norm = LogNorm())
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
#for i in [1,2,3,4,5]:
#    for j in [100,200,300,400,500]:
#        filterAndDrawHeatmap(data, 
#                             heatImage='heat_'+str(i)+'_'+str(j)+'_lognorm.jpg',
#                             heatmapTitle=str(i*35) + 'x' + str(i*45) + ' bins, maximum value is ' + str(j), 
#                             heatMax=j, 
#                             bins=[i*35,i*45])
        #print '\includegraphics[scale=0.35]{heat_'+str(i)+'_'+str(j) + '}'
filterAndDrawHeatmap(data, heatMax = 100, heatImage = "proba.jpg")

# <codecell>

data.describe('OnlyMaleNum', 'OnlyFemaleNum', 'UnisexAndMaleNum', 'UnisexAndFemaleNum','OnlyAccessibleNum').show()

# <codecell>

data.describe('AccessibleOnlyMaleNum', 'AccessibleOnlyFemaleNum', 'AccessibleUnisexAndMaleNum', 'AccessibleUnisexAndFemaleNum').show()

# <codecell>

data.corr('MLAKNum', 'RealAccessibleUnisexNum')

# <codecell>

def facilityTypeToIsCPNum(facilityType):
   if facilityType == 'Park or reserve' or facilityType == 'Camping ground' or facilityType == 'Car park' or facilityType == 'Caravan park': 
        return 1
   else:
         return 0
    
udfFacilityTypeToIsCPNum = udf(facilityTypeToIsCPNum, IntegerType())
onlyCaravanParks = data.filter("FacilityType = 'Park or reserve' or FacilityType == 'Camping ground' or FacilityType == 'Car park' or FacilityType == 'Caravan park'")
onlyCaravanParks.describe('DumpPointNum').show()
onlyWithDumpPoint = data.filter(data.DumpPoint)
onlyWithDumpPoint = onlyWithDumpPoint.withColumn('IsCPNum', udfFacilityTypeToIsCPNum(onlyWithDumpPoint.FacilityType))
onlyWithDumpPoint.describe('IsCPNum').show()

# <codecell>

for row in data.cube('ToiletType').count().collect():
    print "\t\t\item " + str(row.ToiletType)

# <codecell>

data.cube('FacilityType').count().show()

# <codecell>

withAccessible = data.filter(data.AccessibleUnisex | data.AccessibleFemale | data.AccessibleMale)
withAccessible.describe("MLAKNum").show()

# <codecell>

data.cube('ToiletType').count().show()

# <codecell>

withTT = data.filter('ToiletType != null' ).filter('ToiletType != "Sewerage"')

# <codecell>

def toiletTypeToColor(toiletType):
   if toiletType == 'Septic':
        return 'blue'
   elif toiletType == 'Compost':
         return 'green'
   elif toiletType == 'Sewerage':
         return 'yellow'
   elif toiletType == 'Sealed Vault':
         return 'black'
   elif toiletType == 'Pit':
         return 'cyan'
   elif toiletType == 'Drop toilet':
         return 'red'
   elif toiletType == 'Automatic':
         return 'magenta'
    
toiletTypeToColor = udf(toiletTypeToColor, StringType())
withTT = withTT.withColumn('PointColor',toiletTypeToColor(withTT.ToiletType) ); 
colors = [i.PointColor for i in withTT.select('PointColor').collect()]
filterAndDrawScatter(withTT, color = [i.PointColor for i in withTT.select('PointColor').collect()], scatterImage='type_of_toilets_without_sew', 
                     scatterTitle="Type of toilets without Sewerage type")

# <codecell>


