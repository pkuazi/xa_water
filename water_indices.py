from psutil._common import sfan
def get_band():
    NIR = **
    R = **
    B = **
    G = **
    
    return NIR
def ccci():
    CCCI = ((NIR - RE) / (NIR + RE)) / ((NIR - R) / (NIR + R))
    
def ndwi():
    NVWI = (G - NIR) / (G + NIR)
    
def ndwi_lmc():
    import rasterio as rio
    import os
    
    import warnings
    warnings.filterwarnings('ignore')
    
    #setting the env variables 
    os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
    
    #============================Okeechobee============================#
    
    #reading bands
    Okch_R20 = './data/Okeechobee/2A/S2B_MSIL2A_20181222T160509_N0211_R054_T17RNK_20181222T195126.SAFE/GRANULE/L2A_T17RNK_A009375_20181222T160507/IMG_DATA/R20m/'
    b8A = rio.open(Okch_R20 + 'T17RNK_20181222T160509_B8A_20m.jp2')
    meta = b8A.meta
    b8A = b8A.read()
    b11 = rio.open(Okch_R20 + 'T17RNK_20181222T160509_B11_20m.jp2')
    b11 = b11.read()
    
    ndwi_lmc = (b8A.astype(float) - b11.astype(float))/(b8A + b11)
    
    meta.update(driver = 'GTiff')
    meta.update(dtype = rio.float32)
    
    #writing the band as a tiff image
    with rio.open('./data/Okeechobee/2A/ndwi_lmc.tiff', 'w', **meta) as dst:
        dst.write(ndwi_lmc.astype(rio.float32))
    
    
    #============================Chilikha============================#
    
    Chlk_R20 = './data/Chilikha/2A/S2B_MSIL2A_20181224T045219_N0211_R076_T45QTC_20181224T081321.SAFE/GRANULE/L2A_T45QTC_A009397_20181224T050049/IMG_DATA/R20m/'
    b8A = rio.open(Chlk_R20 + 'T45QTC_20181224T045219_B8A_20m.jp2')
    meta = b8A.meta
    b8A = b8A.read()
    b11 = rio.open(Chlk_R20 + 'T45QTC_20181224T045219_B11_20m.jp2')
    b11 = b11.read()
    
    ndwi_lmc = (b8A.astype(float) - b11.astype(float))/(b8A + b11)
    
    meta.update(driver = 'GTiff')
    meta.update(dtype = rio.float32)
    
    #writing the band as a tiff image
    with rio.open('./data/Chilikha/2A/ndwi_lmc.tiff', 'w', **meta) as dst:
        dst.write(ndwi_lmc.astype(rio.float32))
        
def ndwi_wc():
    import rasterio as rio
    import os
    
    import warnings
    warnings.filterwarnings('ignore')
    
    #setting the env variables 
    os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
    
    #============================Okeechobee============================#
    
    #reading bands
    Okch_R10 = './data/Okeechobee/2A/S2B_MSIL2A_20181222T160509_N0211_R054_T17RNK_20181222T195126.SAFE/GRANULE/L2A_T17RNK_A009375_20181222T160507/IMG_DATA/R10m/'
    b3 = rio.open(Okch_R10 + 'T17RNK_20181222T160509_B03_10m.jp2')
    meta = b3.meta
    b3 = b3.read()
    b8 = rio.open(Okch_R10 + 'T17RNK_20181222T160509_B08_10m.jp2')
    b8 = b8.read()
    
    ndwi_wc = (b3.astype(float) - b8.astype(float))/(b3 + b8)
    
    meta.update(driver = 'GTiff')
    meta.update(dtype = rio.float32)
    
    #writing the band as a tiff image
    with rio.open('./data/Okeechobee/2A/ndwi_wc.tiff', 'w', **meta) as dst:
        dst.write(ndwi_wc.astype(rio.float32))
    
    
    #============================Chilikha============================#
    
    Chlk_R10 = './data/Chilikha/2A/S2B_MSIL2A_20181224T045219_N0211_R076_T45QTC_20181224T081321.SAFE/GRANULE/L2A_T45QTC_A009397_20181224T050049/IMG_DATA/R10m/'
    b3 = rio.open(Chlk_R10 + 'T45QTC_20181224T045219_B03_10m.jp2')
    meta = b3.meta
    b3 = b3.read()
    b8 = rio.open(Chlk_R10 + 'T45QTC_20181224T045219_B08_10m.jp2')
    b8 = b8.read()
    
    ndwi_wc = (b3.astype(float) - b8.astype(float))/(b3 + b8)
    
    meta.update(driver = 'GTiff')
    meta.update(dtype = rio.float32)
    
    #writing the band as a tiff image
    with rio.open('./data/Chilikha/2A/ndwi_wc.tiff', 'w', **meta) as dst:
        dst.write(ndwi_wc.astype(rio.float32))
        
def ndci():
    import rasterio as rio
    import os
    
    import warnings
    warnings.filterwarnings('ignore')
    
    #setting the env variables 
    os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
    
    #============================Okeechobee============================#
    
    #reading bands
    Okch_R20 = './data/Okeechobee/2A/S2B_MSIL2A_20181222T160509_N0211_R054_T17RNK_20181222T195126.SAFE/GRANULE/L2A_T17RNK_A009375_20181222T160507/IMG_DATA/R20m/'
    b5 = rio.open(Okch_R20 + 'T17RNK_20181222T160509_B05_20m.jp2')
    meta = b5.meta
    b5 = b5.read()
    b4 = rio.open(Okch_R20 + 'T17RNK_20181222T160509_B04_20m.jp2')
    b4 = b4.read()
    
    ndci = (b5.astype(float) - b4.astype(float))/(b5 + b4)
    chla = (14.039 + 86.115*ndci.astype(float) + 194.325*ndci.astype(float)*ndci.astype(float))
    
    meta.update(driver = 'GTiff')
    meta.update(dtype = rio.float32)
    
    #writing the band as a tiff image
    with rio.open('./data/Okeechobee/2A/ndci.tiff', 'w', **meta) as dst:
        dst.write(ndci.astype(rio.float32))
    
    with rio.open('./data/Okeechobee/2A/chla.tiff', 'w', **meta) as dst:
        dst.write(chla.astype(rio.float32))
    
    
    #============================Chilikha============================#
    
    Chlk_R20 = './data/Chilikha/2A/S2B_MSIL2A_20181224T045219_N0211_R076_T45QTC_20181224T081321.SAFE/GRANULE/L2A_T45QTC_A009397_20181224T050049/IMG_DATA/R20m/'
    b5 = rio.open(Chlk_R20 + 'T45QTC_20181224T045219_B05_20m.jp2')
    meta = b5.meta
    b5 = b5.read()
    b4 = rio.open(Chlk_R20 + 'T45QTC_20181224T045219_B04_20m.jp2')
    b4 = b4.read()
    
    ndci = (b5.astype(float) - b4.astype(float))/(b5 + b4)
    chla = (14.039 + 86.115*ndci.astype(float) + 194.325*ndci.astype(float)*ndci.astype(float))
    
    meta.update(driver = 'GTiff')
    meta.update(dtype = rio.float32)
    
    #writing the band as a tiff image
    # with rio.open('./data/Chilikha/2A/ndci.tiff', 'w', **meta) as dst:
    #     dst.write(ndci.astype(rio.float32))
    
    with rio.open('./data/Chilikha/2A/chla.tiff', 'w', **meta) as dst:
        dst.write(chla.astype(rio.float32))
        
def 