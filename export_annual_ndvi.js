var startYear = 1986;
var endYear = 2023;




var landsat_ds = require('users/jiapeima44/show_temperature:tools/landsat_ds.js');
var landsat_ndvi =landsat_ds.landsat_ndvi.filterBounds(table4)
                                        .filter(ee.Filter.calendarRange(6, 9, 'month'))
                                        .filter(ee.Filter.calendarRange(startYear, endYear, 'year')).mean();

// var firstImage = ee.Image(landsat_ndvi.toList(landsat_ndvi.size()).get(6));


var crs = 'PROJCRS["unknown",BASEGEOGCRS["unknown",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ID["EPSG",6326]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8901]]],CONVERSION["unknown",METHOD["Lambert Conic Conformal (2SP)",ID["EPSG",9802]],PARAMETER["Latitude of false origin",30,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitude of false origin",87,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8822]],PARAMETER["Latitude of 1st standard parallel",30,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",35,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8827]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]' 

var ndviVis = {
  min: -0.1,
  max: 0.5,
  palette: [
    'ffffff', 'ce7e45', 'fcd163', 'c6ca02', '22cc04', '99b718', '207401',
    '012e01'
  ],
};
// print(firstImage);
Map.addLayer(landsat_ndvi, ndviVis);

for (var year = startYear; year <= endYear; year++) {
  // 使用 filterDate 函数筛选当前年份的数据
    var dsNDVI = landsat_ds.landsat_ndvi.filter(ee.Filter.calendarRange(year, year, 'year')).max()

    
    print(dsNDVI);
  // 定义导出任务
    Export.image.toDrive({
      image: dsNDVI, 
      description: 'landsat_ndvi_max_'+ year,
      fileNamePrefix: 'NDVI_MAX' + year, 
      folder: 'ndvi',    
      scale: 30,    
      region: table3,
      maxPixels: 1e10,
      fileFormat: 'GeoTIFF',
    });
    Map.addLayer(dsNDVI, ndviVis, "ndvi"+year)
  
}

