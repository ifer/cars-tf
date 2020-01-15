const tf = require('@tensorflow/tfjs-node');
const fs = require ('fs');

const tools = require('./tools.js');
const log = tools.log;
const show = tools.show;
const showtable = tools.showtable;

const fileUrl =  'file:///home/ifer/dvp/nodejs/cars-tf/data/car_features.csv';

var max_price;
var min_price;
var max_km;
var min_km;
var max_age;
var min_age;


run();
// test();



async function run() {
    const data = await getData();

    // show(values.length);
    // Convert the data to a form we can use for training.
    const tensorData = prepareData (data);
    // const {labels} = tensorData;
    // await labels.forEachAsync(e => console.log(e));

}

async function prepareData (data){
    // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  // return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor after separating ys form xs
        // See comments.txt [3]
        const inputs = await data.map(d => d.xs).toArray();
        const labels = await data.map(d => d.ys).toArray();

        // inputs.map((x) => {x[1] = (x[1] == 'Diesel')? -1 : 1});
        normalizeFuel(inputs);
        normalizeInputs(inputs);
        // console.log(inputs);


        // See comments.txt [4]
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 3]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
        // inputTensor.print();
        // await labels.forEachAsync(e => console.log(e));
        // console.log(labels);
        // inputTensor.print();
        return {
            labels: labelTensor,
        };
  // });
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

    //See comments.txt [1]
    let carsData = tf.data.csv(fileUrl, csvConfig).filter(car => (car.xs.fuel === 'Diesel' || car.xs.fuel === 'Essence') && car.ys.price > 1000);
    // await carsData.forEachAsync(e => console.log(e));


    //See comments.txt [2]
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

    // await flattenedDataset.forEachAsync(e => console.log(e));


    return flattenedDataset;
}

function normalizeFuel(array){
    array.map((x) => {x[1] = (x[1] == 'Diesel')? -1 : 1});
}

function normalizeInputs(array){
    // max_km = tf.maximum();
    min_km;
    max_age;
    min_age;
    show(array[0]);
    // var col1 = array[0].map(function(value,index) { return value[0]; });
    // show(col1);
}
