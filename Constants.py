DATASET = "Foursquare"  # Yelp2018  Gowalla  Foursquare  Yelp
ENCODER = 'hGCN'  # hGCN  Transformer  TransformerLS  gMLP
ABLATION = 'Full'  # Full  w/oImFe  w/oFeTra w/oGlobal w/oAtt w/oConv w/oGraIm
COLD_START = False  # True, False
DEVICE='cuda:1'

DiffSize=1
Beta_min=0.1
Beta_max=20
Stepsize=0.01

user_dict = {
    'ml-1M': 6038,  # 3533
    'douban-book': 12859,
    'Gowalla': 18737,
    'Yelp2018': 31668,
    'Foursquare': 7642,
}

poi_dict = {
    'ml-1M': 3533,  # 3629
    'douban-book': 22294,
    'Gowalla': 32510,
    'Yelp2018': 38048,
    'Foursquare': 28483
}

POI_NUMBER = poi_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)

print('Dataset:', DATASET, '#User:', USER_NUMBER, '#POI', POI_NUMBER)
print('Encoder: ', ENCODER)
print('ABLATION: ', ABLATION)
print('COLD_START: ', COLD_START)

PAD = 0
