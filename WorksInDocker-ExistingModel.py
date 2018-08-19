
# coding: utf-8

# In[ ]:


import pyspark
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType, StringType, DoubleType
from pyspark.ml import Transformer, Estimator, Pipeline
from pyspark.ml.classification import LogisticRegression

import pyspark
spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
            .config("spark.jars.packages", "Azure:mmlspark:0.13") \
            .getOrCreate()
import mmlspark

from mmlspark import CNTKModel, ModelDownloader

from mmlspark import *

import numpy as np, pandas as pd, os, sys, time
from os.path import join, abspath, exists


# In[ ]:


model = ModelDownloader(spark, "models").downloadByName("ResNet50")


# In[ ]:

# must be absolute path if running on single node
image_path = '/notebooks/caltech-256-image-dataset/256_ObjectCategories/*/*.jpg'


# In[ ]:


def getLabel(path):
    if '138.mattress' in path: label = 138.0
    elif '107.hot-air-balloon' in path: label = 107.0
    elif '037.chess-board' in path: label = 37.0
    elif '022.buddha-101' in path: label = 22.0
    elif '024.butterfly' in path: label = 24.0
    elif '135.mailbox' in path: label = 135.0
    elif '083.gas-pump' in path: label = 83.0
    elif '151.ostrich' in path: label = 151.0
    elif '200.stained-glass' in path: label = 200.0
    elif '185.skateboard' in path: label = 185.0
    elif '104.homer-simpson' in path: label = 104.0
    elif '249.yo-yo' in path: label = 249.0
    elif '073.fireworks' in path: label = 73.0
    elif '190.snake' in path: label = 190.0
    elif '222.tombstone' in path: label = 222.0
    elif '005.baseball-glove' in path: label = 5.0
    elif '198.spider' in path: label = 198.0
    elif '039.chopsticks' in path: label = 39.0
    elif '009.bear' in path: label = 9.0
    elif '194.socks' in path: label = 194.0
    elif '049.cormorant' in path: label = 49.0
    elif '121.kangaroo-101' in path: label = 121.0
    elif '041.coffee-mug' in path: label = 41.0
    elif '219.theodolite' in path: label = 219.0
    elif '150.octopus' in path: label = 150.0
    elif '140.menorah-101' in path: label = 140.0
    elif '243.welding-mask' in path: label = 243.0
    elif '001.ak47' in path: label = 1.0
    elif '061.dumb-bell' in path: label = 61.0
    elif '127.laptop-101' in path: label = 127.0
    elif '195.soda-can' in path: label = 195.0
    elif '045.computer-keyboard' in path: label = 45.0
    elif '213.teddy-bear' in path: label = 213.0
    elif '034.centipede' in path: label = 34.0
    elif '174.rotary-phone' in path: label = 174.0
    elif '161.photocopier' in path: label = 161.0
    elif '237.vcr' in path: label = 237.0
    elif '010.beer-mug' in path: label = 10.0
    elif '178.school-bus' in path: label = 178.0
    elif '019.boxing-glove' in path: label = 19.0
    elif '131.lightbulb' in path: label = 131.0
    elif '236.unicorn' in path: label = 236.0
    elif '063.electric-guitar-101' in path: label = 63.0
    elif '047.computer-mouse' in path: label = 47.0
    elif '089.goose' in path: label = 89.0
    elif '204.sunflower-101' in path: label = 204.0
    elif '017.bowling-ball' in path: label = 17.0
    elif '248.yarmulke' in path: label = 248.0
    elif '113.hummingbird' in path: label = 113.0
    elif '053.desk-globe' in path: label = 53.0
    elif '183.sextant' in path: label = 183.0
    elif '026.cake' in path: label = 26.0
    elif '079.frisbee' in path: label = 79.0
    elif '025.cactus' in path: label = 25.0
    elif '214.teepee' in path: label = 214.0
    elif '142.microwave' in path: label = 142.0
    elif '092.grapes' in path: label = 92.0
    elif '229.tricycle' in path: label = 229.0
    elif '108.hot-dog' in path: label = 108.0
    elif '252.car-side-101' in path: label = 252.0
    elif '077.french-horn' in path: label = 77.0
    elif '235.umbrella-101' in path: label = 235.0
    elif '095.hamburger' in path: label = 95.0
    elif '242.watermelon' in path: label = 242.0
    elif '003.backpack' in path: label = 3.0
    elif '188.smokestack' in path: label = 188.0
    elif '231.tripod' in path: label = 231.0
    elif '187.skyscraper' in path: label = 187.0
    elif '011.billiards' in path: label = 11.0
    elif '218.tennis-racket' in path: label = 218.0
    elif '168.raccoon' in path: label = 168.0
    elif '184.sheet-music' in path: label = 184.0
    elif '058.doorknob' in path: label = 58.0
    elif '055.dice' in path: label = 55.0
    elif '086.golden-gate-bridge' in path: label = 86.0
    elif '148.mussels' in path: label = 148.0
    elif '066.ewer-101' in path: label = 66.0
    elif '217.tennis-court' in path: label = 217.0
    elif '038.chimp' in path: label = 38.0
    elif '012.binoculars' in path: label = 12.0
    elif '093.grasshopper' in path: label = 93.0
    elif '226.traffic-light' in path: label = 226.0
    elif '054.diamond-ring' in path: label = 54.0
    elif '099.harpsichord' in path: label = 99.0
    elif '015.bonsai-101' in path: label = 15.0
    elif '146.mountain-bike' in path: label = 146.0
    elif '221.tomato' in path: label = 221.0
    elif '159.people' in path: label = 159.0
    elif '196.spaghetti' in path: label = 196.0
    elif '257.clutter' in path: label = 257.0
    elif '101.head-phones' in path: label = 101.0
    elif '097.harmonica' in path: label = 97.0
    elif '084.giraffe' in path: label = 84.0
    elif '059.drinking-straw' in path: label = 59.0
    elif '062.eiffel-tower' in path: label = 62.0
    elif '251.airplanes-101' in path: label = 251.0
    elif '078.fried-egg' in path: label = 78.0
    elif '227.treadmill' in path: label = 227.0
    elif '256.toad' in path: label = 256.0
    elif '207.swan' in path: label = 207.0
    elif '111.house-fly' in path: label = 111.0
    elif '070.fire-extinguisher' in path: label = 70.0
    elif '115.ice-cream-cone' in path: label = 115.0
    elif '245.windmill' in path: label = 245.0
    elif '175.roulette-wheel' in path: label = 175.0
    elif '255.tennis-shoes' in path: label = 255.0
    elif '130.license-plate' in path: label = 130.0
    elif '122.kayak' in path: label = 122.0
    elif '133.lightning' in path: label = 133.0
    elif '220.toaster' in path: label = 220.0
    elif '162.picnic-table' in path: label = 162.0
    elif '211.tambourine' in path: label = 211.0
    elif '132.light-house' in path: label = 132.0
    elif '069.fighter-jet' in path: label = 69.0
    elif '210.syringe' in path: label = 210.0
    elif '171.refrigerator' in path: label = 171.0
    elif '006.basketball-hoop' in path: label = 6.0
    elif '065.elk' in path: label = 65.0
    elif '147.mushroom' in path: label = 147.0
    elif '050.covered-wagon' in path: label = 50.0
    elif '139.megaphone' in path: label = 139.0
    elif '118.iris' in path: label = 118.0
    elif '212.teapot' in path: label = 212.0
    elif '193.soccer-ball' in path: label = 193.0
    elif '166.praying-mantis' in path: label = 166.0
    elif '030.canoe' in path: label = 30.0
    elif '182.self-propelled-lawn-mower' in path: label = 182.0
    elif '027.calculator' in path: label = 27.0
    elif '106.horseshoe-crab' in path: label = 106.0
    elif '081.frying-pan' in path: label = 81.0
    elif '181.segway' in path: label = 181.0
    elif '076.football-helmet' in path: label = 76.0
    elif '060.duck' in path: label = 60.0
    elif '215.telephone-box' in path: label = 215.0
    elif '105.horse' in path: label = 105.0
    elif '075.floppy-disk' in path: label = 75.0
    elif '007.bat' in path: label = 7.0
    elif '232.t-shirt' in path: label = 232.0
    elif '014.blimp' in path: label = 14.0
    elif '016.boom-box' in path: label = 16.0
    elif '023.bulldozer' in path: label = 23.0
    elif '100.hawksbill-101' in path: label = 100.0
    elif '126.ladder' in path: label = 126.0
    elif '247.xylophone' in path: label = 247.0
    elif '109.hot-tub' in path: label = 109.0
    elif '203.stirrups' in path: label = 203.0
    elif '180.screwdriver' in path: label = 180.0
    elif '155.paperclip' in path: label = 155.0
    elif '129.leopards-101' in path: label = 129.0
    elif '042.coffin' in path: label = 42.0
    elif '169.radio-telescope' in path: label = 169.0
    elif '165.pram' in path: label = 165.0
    elif '051.cowboy-hat' in path: label = 51.0
    elif '029.cannon' in path: label = 29.0
    elif '067.eyeglasses' in path: label = 67.0
    elif '144.minotaur' in path: label = 144.0
    elif '163.playing-card' in path: label = 163.0
    elif '028.camel' in path: label = 28.0
    elif '031.car-tire' in path: label = 31.0
    elif '238.video-projector' in path: label = 238.0
    elif '244.wheelbarrow' in path: label = 244.0
    elif '072.fire-truck' in path: label = 72.0
    elif '123.ketch-101' in path: label = 123.0
    elif '082.galaxy' in path: label = 82.0
    elif '035.cereal-box' in path: label = 35.0
    elif '205.superman' in path: label = 205.0
    elif '172.revolver-101' in path: label = 172.0
    elif '008.bathtub' in path: label = 8.0
    elif '064.elephant-101' in path: label = 64.0
    elif '137.mars' in path: label = 137.0
    elif '199.spoon' in path: label = 199.0
    elif '043.coin' in path: label = 43.0
    elif '125.knife' in path: label = 125.0
    elif '164.porcupine' in path: label = 164.0
    elif '145.motorbikes-101' in path: label = 145.0
    elif '004.baseball-bat' in path: label = 4.0
    elif '186.skunk' in path: label = 186.0
    elif '094.guitar-pick' in path: label = 94.0
    elif '176.saddle' in path: label = 176.0
    elif '154.palm-tree' in path: label = 154.0
    elif '136.mandolin' in path: label = 136.0
    elif '098.harp' in path: label = 98.0
    elif '096.hammock' in path: label = 96.0
    elif '036.chandelier-101' in path: label = 36.0
    elif '167.pyramid' in path: label = 167.0
    elif '224.touring-bike' in path: label = 224.0
    elif '241.waterfall' in path: label = 241.0
    elif '040.cockroach' in path: label = 40.0
    elif '046.computer-monitor' in path: label = 46.0
    elif '240.watch-101' in path: label = 240.0
    elif '254.greyhound' in path: label = 254.0
    elif '192.snowmobile' in path: label = 192.0
    elif '158.penguin' in path: label = 158.0
    elif '173.rifle' in path: label = 173.0
    elif '230.trilobite-101' in path: label = 230.0
    elif '033.cd' in path: label = 33.0
    elif '157.pci-card' in path: label = 157.0
    elif '052.crab-101' in path: label = 52.0
    elif '013.birdbath' in path: label = 13.0
    elif '149.necktie' in path: label = 149.0
    elif '206.sushi' in path: label = 206.0
    elif '179.scorpion-101' in path: label = 179.0
    elif '068.fern' in path: label = 68.0
    elif '048.conch' in path: label = 48.0
    elif '208.swiss-army-knife' in path: label = 208.0
    elif '074.flashlight' in path: label = 74.0
    elif '128.lathe' in path: label = 128.0
    elif '253.faces-easy-101' in path: label = 253.0
    elif '201.starfish-101' in path: label = 201.0
    elif '209.sword' in path: label = 209.0
    elif '091.grand-piano-101' in path: label = 91.0
    elif '225.tower-pisa' in path: label = 225.0
    elif '110.hourglass' in path: label = 110.0
    elif '056.dog' in path: label = 56.0
    elif '002.american-flag' in path: label = 2.0
    elif '088.golf-ball' in path: label = 88.0
    elif '152.owl' in path: label = 152.0
    elif '170.rainbow' in path: label = 170.0
    elif '160.pez-dispenser' in path: label = 160.0
    elif '116.iguana' in path: label = 116.0
    elif '114.ibis-101' in path: label = 114.0
    elif '102.helicopter-101' in path: label = 102.0
    elif '246.wine-bottle' in path: label = 246.0
    elif '191.sneaker' in path: label = 191.0
    elif '057.dolphin-101' in path: label = 57.0
    elif '032.cartman' in path: label = 32.0
    elif '216.tennis-ball' in path: label = 216.0
    elif '153.palm-pilot' in path: label = 153.0
    elif '234.tweezer' in path: label = 234.0
    elif '189.snail' in path: label = 189.0
    elif '156.paper-shredder' in path: label = 156.0
    elif '197.speed-boat' in path: label = 197.0
    elif '020.brain-101' in path: label = 20.0
    elif '090.gorilla' in path: label = 90.0
    elif '233.tuning-fork' in path: label = 233.0
    elif '085.goat' in path: label = 85.0
    elif '117.ipod' in path: label = 117.0
    elif '250.zebra' in path: label = 250.0
    elif '044.comet' in path: label = 44.0
    elif '080.frog' in path: label = 80.0
    elif '228.triceratops' in path: label = 228.0
    elif '143.minaret' in path: label = 143.0
    elif '124.killer-whale' in path: label = 124.0
    elif '021.breadmaker' in path: label = 21.0
    elif '103.hibiscus' in path: label = 103.0
    elif '018.bowling-pin' in path: label = 18.0
    elif '177.saturn' in path: label = 177.0
    elif '112.human-skeleton' in path: label = 112.0
    elif '120.joy-stick' in path: label = 120.0
    elif '134.llama-101' in path: label = 134.0
    elif '239.washing-machine' in path: label = 239.0
    elif '141.microscope' in path: label = 141.0
    elif '202.steering-wheel' in path: label = 202.0
    elif '119.jesus-christ' in path: label = 119.0
    elif '223.top-hat' in path: label = 223.0
    elif '071.fire-hydrant' in path: label = 71.0
    elif '087.goldfish' in path: label = 87.0    
    else:
        label = 300
    return label


# In[ ]:


#Read in images
imageDF = spark.readImages(image_path, sampleRatio=0.005)
getLabelUDF = udf(lambda row: getLabel(row[0]), DoubleType())
imageDF = imageDF.withColumn("labels", getLabelUDF(col('image')))

imageDF.printSchema()


# In[ ]:


# Make some featurizers
it = (ImageTransformer()
    .setOutputCol("scaled")
    .resize(height = 256, width = 256))

ur = (UnrollImage()
    .setInputCol("scaled")
    .setOutputCol("features"))
    
dc1 = DropColumns().setCols(["scaled", "image"])

lr1 = LogisticRegression().setFeaturesCol("features").setLabelCol("labels")

dc2 = DropColumns().setCols(["features"])

basicModel = Pipeline(stages=[it, ur, dc1, lr1, dc2])


# In[ ]:


resnet = (ImageFeaturizer()
    .setInputCol("image")
    .setOutputCol("features")
    .setModelLocation(model.uri)
    .setLayerNames(model.layerNames)
    .setCutOutputLayers(1))
    
dc3 = DropColumns().setCols(["image"])
    
lr2 = LogisticRegression().setFeaturesCol("features").setLabelCol("labels")

dc4 = DropColumns().setCols(["features"])

deepModel = Pipeline(stages=[resnet, dc3, lr2, dc4]) 


# In[ ]:


def timedExperiment(model, train, test):
    start = time.time()
    result =  model.fit(train).transform(test).toPandas()
    print("Experiment took {}s".format(time.time() - start))
    return result


# In[ ]:


train, test = imageDF.randomSplit([.8,.2])
train.count(), test.count()


# In[ ]:


#basicResults = timedExperiment(basicModel, train, test)


# In[ ]:


deepResults = timedExperiment(deepModel, train, test)

