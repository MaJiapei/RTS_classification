
var cloudMask457 = function(image) {
  var qa = image.select('pixel_qa');
  // If the cloud bit (5) is set and the cloud confidence (7) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 5)
                  .and(qa.bitwiseAnd(1 << 7))
                  .or(qa.bitwiseAnd(1 << 3));
  // Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};



var cloudMask8 = function (image) {
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask);
};


var clearMask = function(image) {
  var qa = image.select('pixel_qa');
  var clearBitMask = qa.bitwiseAnd(1 << 6);
  return image.updateMask(clearBitMask);
};


var coefficients = {
  itcps: ee.Image.constant([0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]),
  slopes: ee.Image.constant([0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071])
};

// Define function to get and rename bands of interest from OLI.
function rename_oli(img) {
  return img.select(
      ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'],
      ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa']);
}

// Define function to get and rename bands of interest from ETM+.
function rename_tm(img) {
  return img.select(
      ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'],
      ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa']);
}


function harmonize_tm_oli(img) {
  return img.select(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'])
      .multiply(coefficients.slopes)
      .add(coefficients.itcps)
      .addBands(img.select('pixel_qa'))
      .copyProperties(img)
      .copyProperties(img, ["system:time_start", "system:id"]);
}


function cast(img) {
  return img.cast({"Blue": ee.PixelType('float', -0.2, 1.6),
                   "Green":  ee.PixelType('float', -0.2, 1.6),
                   "Red":  ee.PixelType('float', -0.2, 1.6),
                   "NIR":  ee.PixelType('float', -0.2, 1.6),
                   "SWIR1":  ee.PixelType('float', -0.2, 1.6),
                   "SWIR2":  ee.PixelType('float', -0.2, 1.6)});
}


function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}


var landsat457 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2')
                  .merge(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'))
                  .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'))
                  .map(applyScaleFactors)
                  .map(rename_tm)
                  .map(harmonize_tm_oli)
                  .map(cast);

                  
var landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .map(applyScaleFactors)
                  .map(rename_oli)
                  .map(cast);
                  


var landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                  .map(applyScaleFactors)
                  .map(rename_oli)
                  .map(cast);


var landsat_multi = landsat457.merge(landsat8).merge(landsat9).sort("system:time_start");


var landsat_ndvi = landsat_multi.map(clearMask)
                    .map(function(image){
                          var ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI');
                          return image.addBands(ndvi)
                                      .copyProperties(image, ["system:time_start"]);
                          
                        })
                    .select('NDVI');


var landsat_ndmi = landsat_multi.map(clearMask)
                    .map(function(image){
                          var ndwi = image.normalizedDifference(['Green', 'NIR']).rename('NDMI');
                          return image.addBands(ndwi)
                                      .copyProperties(image, ["system:time_start"]);
                          
                        })
                    .select('NDMI');
                    
var landsat_ndwi = landsat_multi.map(clearMask)
                    .map(function(image){
                          var ndwi = image.normalizedDifference(['SWIR1', 'SWIR2']).rename('NDMI');
                          return image.addBands(ndwi)
                                      .copyProperties(image, ["system:time_start"]);
                          
                        })
                    .select('NDMI');
                    
 
// Step1: 250m to 1000m
var landsat_ndvi_reduce = landsat_ndvi.map(function(image){
      return image
          .reduceResolution({
            reducer: ee.Reducer.mean(),
            maxPixels: 200*200})
          // Request the data at the scale and projection of the MODIS image.
          .reproject({
            crs: image.projection(),
            scale: 4000
          })
  
});

  


exports = {
  cloudMask457: cloudMask457,
  cloudMask8: cloudMask8,
  clearMask: clearMask,
  landsat457: landsat457,
  landsat8: landsat8,
  landsat9: landsat9,
  landsat_multi: landsat_multi,
  landsat_ndvi: landsat_ndvi,
  landsat_ndvi_reduce: landsat_ndvi_reduce,
  landsat_ndmi: landsat_ndmi,
  applyScaleFactors: applyScaleFactors
}
