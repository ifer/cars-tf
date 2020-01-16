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

    // await data.forEachAsync(e => console.log(e));

    // show(values.length);
    // Convert the data to a form we can use for training.
    let [trainingInputs, trainingLabels, testingInputs, testingLabels] = await prepareData (data);
    // inputs.print();
    // labels.print();
    let model = createModel();
    model.summary();
    //
    result = await trainModel(model, trainingInputs, trainingLabels);
    // console.log(result);

    testModel(model, testingInputs, testingLabels);
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single hidden layer with 10 nodes, expecting 3 inputs and having bias
  model.add(tf.layers.dense({inputShape: [3], units: 10, useBias: true}));

  // Add an output layer with bias
  model.add(tf.layers.dense({units: 1, useBias: true}));

  const loss = "meanSquaredError"; // Selected loss function: Mean Squared Error (because the data will be normalized )

  //Optimizer which is going to find minimum loss
  /* const optimizer = tf.train.sgd(0.1); // Selected optimizer: Stochastic Gradient Descent with learning rate 0.1 */
  const optimizer = tf.train.adam(); // Preferred optimizer: the most recently developed and the most efficient

  //// Compile model //// (necessary before training)
  model.compile ({
      loss: loss,
      optimizer,
  });

  return model;
}


function trainModel(model, trainingInputTensor, trainingLabelTensor){



    var t0, t1; // Time holders

    //Training method: fit (returns a promise)
    return model.fit (trainingInputTensor, trainingLabelTensor,{
        batchsize: 32, // Number of samples per gradient update
        epochs: 10, //larger number of iterations on a non-linear model
        validationSplit: 0.2, // fraction of the training data to be used as validation data.
                              // The model will set apart this fraction of the training data, will not train on it,
                              // and will evaluate the loss and any model metrics on this data at the end of each epoch
        callbacks: {
            //
            onTrainBegin: function (logs) {
                                    t0 = getCurrentTime();
                                    console.log ("TRAINING STARTED");
                                },
            onTrainEnd:   function (logs) {
                                    t1 = getCurrentTime();
                                    let timeElapsed = ((t1 - t0) / 60000).toFixed(2); //in minutes
                                    console.log (`TRAINING FINISHED in ${timeElapsed} minutes`);
                                },

            // On end of each epoch and of batch, print epoch number and loss (visual version)
            // onEpochEnd,
            // On end of each epoch print epoch number and loss (console version)
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`),

            onEpochBegin: async function () {
            }
        }
    });
}

async function testModel(model, testingInputTensor, testingLabelTensor){
    //Testing the model
    const lossTensor = model.evaluate(testingInputTensor, testingLabelTensor); // we get a tensor
    const loss = (await lossTensor.dataSync())[0]; // Get a scalar value
    console.log(`Testing set loss = ${loss}`);

    //Update status
    // document.getElementById("testing-status").innerHTML = `Testing set loss: ${loss.toPrecision(5)}`;
}

async function prepareData (data){

        // console.log(data);

        // Step 1. Shuffle the data and convert dataset to array
        let dataArray = await data.shuffle(1000).toArray();
        // await dataArray.forEachAsync(e => console.log(e));


        //Use 80% of data for training and 20% for testing
        let [trainingSet, testingSet] = splitData(dataArray, 0.8);
        // show(`training=${trainingSet.length} testing=${testingSet.length} sum=${dataArray.length}`);

        // Step 2. Convert data to Tensor after separating ys form xs
        // See comments.txt [3]
        const trainingInputs = trainingSet.map(d => d.xs);
        const trainingLabels = trainingSet.map(d => d.ys);

        const testingInputs = testingSet.map(d => d.xs);
        const testingLabels = testingSet.map(d => d.ys);
        // const inputs = await data.map(d => d.xs).toArray();
        // const labels = await data.map(d => d.ys).toArray();

        //Normalize fuel before converting to tensor to get rid of string values
        normalizeFuel(trainingInputs);
        normalizeFuel(testingInputs);

        // See comments.txt [4]
        let trainingInputTensor = tf.tensor2d(trainingInputs);
        let testingInputTensor = tf.tensor2d(testingInputs);

        let normalizedTrainingInputs = tf.tidy(() => normalizeInputs(trainingInputTensor));
        let normalizedTestingInputs = tf.tidy(() => normalizeInputs(testingInputTensor));
        // normalizedTrainingInputs.print();
        // return;

        let trainingLabelTensor = tf.tensor2d(trainingLabels, [trainingLabels.length, 1]);
        let testingLabelTensor = tf.tensor2d(testingLabels, [testingLabels.length, 1]);

        let normalizedTrainingLabels = normalizeLabels(trainingLabelTensor);
        let normalizedTestingLabels = normalizeLabels(testingLabelTensor);
        // normalizedTrainingLabels.print();

        trainingInputTensor.dispose();
        testingInputTensor.dispose();
        trainingLabelTensor.dispose();
        testingLabelTensor.dispose();

        return [normalizedTrainingInputs, normalizedTrainingLabels,
                normalizedTestingInputs, normalizedTestingLabels];

}


//Split data into training set and test set according to threshold value
function splitData(dataArray, threshold){
    let rows = dataArray.length;
    let trainRows = Math.floor(rows * threshold);
    let trainingSet = dataArray.slice(0, trainRows);
    let testingSet = dataArray.slice(trainRows);

    return [trainingSet, testingSet];
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

function getCurrentTime(){
    let d = new Date();
    let t = d.getTime();
    return (t);
}
