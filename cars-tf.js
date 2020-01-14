const tf = require('@tensorflow/tfjs-node');
const fs = require ('fs');

const tools = require('./tools.js');
const log = tools.log;
const show = tools.show;
const showtable = tools.showtable;

const fileUrl =  'file:///home/ifer/dvp/nodejs/cars-tf/data/car_features.csv';

run();
// test();



async function run() {
    const data = await getData();

    // show(values.length);

}

async function getData() {
    let csvConfig = {
        hasHeader: true,
        columnConfigs: {
            price: {
                isLabel: true
            }
        }
    };

    let carsData = tf.data.csv(fileUrl, csvConfig);
    // await carsData.forEachAsync(e => console.log(e));

    const flattenedDataset =
        carsData.map(({
            xs,
            ys
        }) => {
            // Convert xs(features) and ys(labels) from object form (keyed by
            // column name) to array form.
            return {
                xs: Object.values(xs),
                ys: Object.values(ys)
            };
        });

    await flattenedDataset.forEachAsync(e => console.log(e));


    return flattenedDataset;
}
