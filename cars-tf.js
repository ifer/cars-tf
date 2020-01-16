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
    let [labels, inputs] = await prepareData (data);
    inputs.print();
    labels.print();

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

        //Normalize fuel before converting to tensor to get rid of string values
        normalizeFuel(inputs);

        // See comments.txt [4]
        let inputTensor = tf.tensor2d(inputs);

        let normalizedInputs = tf.tidy(() => normalizeInputs(inputTensor));
        // normalizedInputs.print();

        let labelTensor = tf.tensor2d(labels, [labels.length, 1]);
        let normalizedLabels = normalizeLabels(labelTensor);
        // normalizedLabels.print();


        return [normalizedLabels, normalizedInputs];

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

function normalizeInputs(inputTensor){
    // Split into 3 separate tensors - one of each input feature
    let [km, fuel, age] = tf.split(inputTensor, 3, 1); // split on axis=1 i.e. vertically
    //Find min and max of 1st and 3rd tensors
    max_km = km.max();
    min_km = km.min();
    max_age = age.max();
    min_age = age.min();

    //Normalize 1st and 3rd tensors
    let normalizedKm = km.sub(min_km).div(max_km.sub(min_km)); // (value - min) / (max - min)
    let normalizedAge = age.sub(min_age).div(max_age.sub(min_age)); // (value - min) / (max - min)

    //Concatenate again the three tensors according to dimension 1 ("vertically")
    let normalizedInputTensor = tf.concat([normalizedKm, fuel, normalizedAge ], 1);
    // normalizedInputTensor.print();
    return normalizedInputTensor;

}

function normalizeLabels(labelTensor){
    max_price = labelTensor.max();
    min_price = labelTensor.min();

    let normalizedLabelTensor = labelTensor.sub(min_price).div(max_price.sub(min_price)); // (value - min) / (max - min)

    return normalizedLabelTensor;
}
