import os,ogr, osr
import fiona
from shapely.geometry import mapping, Polygon

def wkt_to_shp(coords, epsg):
    geom = ogr.CreateGeometryFromWkt(wkt_str)
    minx_wgs, maxx_wgs, miny_wgs, maxy_wgs = geom.GetEnvelope()
    
    print(minx_wgs, maxx_wgs, miny_wgs, maxy_wgs)
    
    drv = ogr.GetDriverByName("ESRI Shapefile")
    ds = drv.CreateDataSource( '/mnt/win/data/xiongan.shp' )
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
                     
    layer = ds.CreateLayer('boundary' , srs = srs )
    fieldDefn = ogr.FieldDefn('id', ogr.OFTInteger)
    
    layer.CreateField(fieldDefn)
    featureDefn=layer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(geom)
    feature.SetField('id',1)     
    layer.CreateFeature(feature)
    

coords_path = '/mnt/win/data/xiongan/xiongan.txt'
coords = open(coords_path,'r')
c_str = coords.read()
coords_l = list(c_str.split(';'))
# wkt_p = geom_wkt = 'POLYGON ((%s %s,%s %s,%s %s,%s %s,%s %s))' % (ltx_wgs, lty_wgs, rtx_wgs, rty_wgs, rbx_wgs, rby_wgs, lbx_wgs, lby_wgs, ltx_wgs, lty_wgs)
wkt_str = 'POLYGON (('
for ll in coords_l:
    x,y = ll.split(',')
    c = x+' '+y+','
    wkt_str+=c
wkt_str = wkt_str[:-1]+'))'
# wkt_to_shp(wkt_str, 4326)

geom = ogr.CreateGeometryFromWkt(wkt_str)
minx_wgs, maxx_wgs, miny_wgs, maxy_wgs = geom.GetEnvelope()
schema={'geometry': 'Polygon', 'properties': {'id': 'int'} }
#  use fiona.open
with fiona.open('/mnt/win/data/xiongan/xaenv.shp', mode='w', driver='ESRI Shapefile', schema=schema, crs='EPSG:4326', encoding='utf-8') as layer:
    poly=Polygon([[minx_wgs,maxy_wgs],[maxx_wgs,maxy_wgs],[maxx_wgs,miny_wgs],[minx_wgs,miny_wgs],[minx_wgs,maxy_wgs]])
    element = {'geometry':mapping(poly), 'properties': {'id': 1}}
    layer.write(element)     
          