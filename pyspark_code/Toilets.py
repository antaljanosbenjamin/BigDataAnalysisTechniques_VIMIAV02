# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
import pyspark.sql.types
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql import SQLContext
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import pandas
import numpy as np

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
            print 'corr(' + col1 + ', ' + col2 + ') = ' + str(correlation) 

# <codecell>

def initializePlt (ranges, title):
    plt.clf()
    plt.rcParams.update({'font.size' : 14})
    plt.title(title, fontsize = 18)
    plt.axis(ranges)
    
def transformDatas(datas, filter):
    coords = datas.filter(filter).select('Longitude','Latitude')
    panda = coords.toPandas()
    values = panda.get_values()
    x,y = zip(*values)
    return x,y    
    
def filterAndDrawScatter(datas, *args, **kwargs):
    
    alpha = kwargs.get('alpha',1.0)
    marker = kwargs.get('marker', 'o')
    ranges = kwargs.get('ranges', [110,155,-45,-10])
    title = kwargs.get('scatterTitle',' ')
    color = kwargs.get('color', 'blue')
    
    initializePlt(ranges,title)
    
    x, y = transformDatas(datas,kwargs.get('filterString', 'True'))
    
    plt.scatter(x,y, edgecolors="none", alpha = alpha, color = color)
    scatterImage = kwargs.get('scatterImage',None)
    if (scatterImage is not None):
        plt.savefig(scatterImage, bbox_inches = 'tight', pad_inches = 0)
    if kwargs.get('showScatter',True):
        plt.show()
    return

def filterAndDrawHeatmap(datas, *args, **kwargs):
    heatMax = kwargs.get('heatMax', 30)
    ranges = kwargs.get('ranges', [110,155,-45,-10])
    x, y = transformDatas(datas,kwargs.get('filterString', 'True'))
    
    initializePlt(ranges,kwargs.get('heatmapTitle',''))
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=kwargs.get('bins',[140,180]), range = [[ranges[0], ranges[1]], [ranges[2], ranges[3]]])
    for i in range(len(heatmap)):
            for j in range(len(heatmap[i])):
                    if heatmap[i][j] > heatMax:
                        heatmap[i][j] = heatMax
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.title(kwargs.get('heatmapTitle',' '))
    imgplot = plt.imshow(heatmap.T, extent=extent, origin='lower', norm = LogNorm())
    plt.colorbar()
    heatImage = kwargs.get('heatImage',None)
    if (heatImage is not None):
        plt.savefig(heatImage, bbox_inches = 'tight', pad_inches = 0)
    return

# <codecell>

for i in [1,2,3,4,5]:
    for j in [100,200,300,400,500]:
        filterAndDrawHeatmap(data, 
                             heatImage='heat_'+str(i)+'_'+str(j)+'_lognorm.jpg',
                             heatmapTitle=str(i*35) + 'x' + str(i*45) + ' bins, maximum value is ' + str(j), 
                             heatMax=j, 
                             bins=[i*35,i*45])
        #print '\includegraphics[scale=0.35]{heat_'+str(i)+'_'+str(j) + '}'

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

data.cube('FacilityType').count().show()
data.cube('ToiletType').count().show()

# <codecell>

withToiletType = data.filter('ToiletType != null')

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
withToiletType = withToiletType.withColumn('PointColor',toiletTypeToColor(withToiletType.ToiletType) ); 
colors = [i.PointColor for i in withToiletType.select('PointColor').collect()]
filterAndDrawScatter(withToiletType, color = colors, scatterImage='type_of_toilets', 
                     scatterTitle="Type of toilets")
withoutSew = withToiletType.filter("ToiletType != 'Sewerage'")
colors = [i.PointColor for i in withoutSew.select('PointColor').collect()]
filterAndDrawScatter(withoutSew, color = colors, scatterImage='type_of_toilets_without_sew', 
                     scatterTitle="Type of toilets without sewerage")
withoutSewAndSep = withoutSew.filter("ToiletType != 'Septic'")
colors = [i.PointColor for i in withoutSewAndSep.select('PointColor').collect()]
filterAndDrawScatter(withoutSewAndSep, color = colors, scatterImage='type_of_toilets_without_sew_and_sep', 
                     scatterTitle="Type of toilets without sewerage and septic")

# <codecell>

inShoppingCenters = withToiletType.filter("FacilityType = 'Shopping centre'")

# <codecell>

filterAndDrawScatter(inShoppingCenters, scatterImage='in_shopping_centers', scatterTitle='Toilets in shopping centers')

# <codecell>

print "Number of toilets with Female, Male and Unisex type: " + str(data.filter("Female and Male and Unisex").count())

# <codecell>

print "Number of accessible toilets with Female, Male and Unisex type: " + str(data.filter("AccessibleFemale and AccessibleMale and AccessibleUnisex").count())

# <codecell>

print "Number of toilets with only one of Male/Female type: " + str(data.filter("(Female and not Male) or (not Female and Male)").count())

