const tf = require('@tensorflow/tfjs-node');
const fs = require ('fs');
const yargs = require ('yargs');

const tools = require('./tools.js');
const log = tools.log;
const show = tools.show;
const showtable = tools.showtable;

// const dataFileUrl =  'file:///home/ifer/dvp/nodejs/cars-tf/data/car_features.csv';
const dataFileUrl =  'file://data/car_features.csv';
const modelSaveFileUrl =  'file://saved/car_model';
const modelLoadFileUrl =  'file://saved/car_model/model.json';
const metadataFileUrl =  'saved/car_metadata.json';

let metadata = {};

//Syntax: node cars-tf: creates, trains, tests and saves a model
//        node cars-tf load: loads the last saved model and runs predictions

const argv = yargs
             .command ('load', 'Run a saved model')
             .help()
             .argv;

 var command = argv._[0];



if (command === 'load'){
     runSavedModel();
}
 else {
    run();
 }

async function runSavedModel() {
    model = await loadModel();
    await runPredictions(model);
}


async function run() {


    const data = await getData();
    // await data.forEachAsync(e => console.log(e));

    // Convert the data to a form we can use for training.
    let [trainingInputs, trainingLabels, testingInputs, testingLabels] = await prepareData (data);
    // trainingInputs.print();
    // trainingLabels.print();


    let model = createModel();
    model.summary();
    //
    result = await trainModel(model, trainingInputs, trainingLabels);
    // console.log(result);

    testModel(model, testingInputs, testingLabels);

    // saveModel(model);

    await runPredictions(model);
}

async function getData() {
    let csvConfig = {
        hasHeader: true,
        configuredColumnsOnly: true,
        columnConfigs: {
            price: {
                isLabel: true,
                required: true
            },
            km: {
                required: true
            },
            age: {
                required: true
            }
        }
    };

    //See comments.txt [1]
    let carsData = tf.data.csv(dataFileUrl, csvConfig).filter(car => ( car.ys.price > 1000));

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

        // See comments.txt [4]
        let trainingInputTensor = tf.tensor2d(trainingInputs);
        let testingInputTensor = tf.tensor2d(testingInputs);

        // let normalizedTrainingInputs = tf.tidy(() => normalizeInputs(trainingInputTensor, 'training'));
        // let normalizedTestingInputs = tf.tidy(() => normalizeInputs(testingInputTensor));

        let normalizedTrainingInputs = normalizeInputs(trainingInputTensor, 'training');
        let normalizedTestingInputs = normalizeInputs(testingInputTensor);
        // normalizedTrainingInputs.print();

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


function createModel() {
    let learningRate = 0.1;
  // Create a sequential model
  const model = tf.sequential();

  // Add a single hidden layer with 10 nodes, expecting 3 inputs and having bias
  model.add(tf.layers.dense({
      inputShape: [2],
      units: 10,
      activation: "sigmoid", // non-linear activation function
      useBias: true
  }));

  // Add an output layer with bias
  model.add(tf.layers.dense({
      units: 1,
      activation: "sigmoid", // non-linear activation function
      useBias: true
  }));

  const loss = "meanSquaredError"; // Selected loss function: Mean Squared Error (because the data will be normalized )

  //Optimizer which is going to find minimum loss
  /* const optimizer = tf.train.sgd(0.1); // Selected optimizer: Stochastic Gradient Descent with learning rate 0.1 */
  const optimizer = tf.train.adam(learningRate); // Preferred optimizer: the most recently developed and the most efficient

  //// Compile model //// (necessary before training)
  model.compile ({
      loss: loss,
      optimizer: optimizer,
      metrics: ['accuracy'],
  });

  return model;
}


function trainModel(model, trainingInputTensor, trainingLabelTensor){



    var t0, t1; // Time holders

    //Training method: fit (returns a promise)
    return model.fit (trainingInputTensor, trainingLabelTensor,{
        batchsize: 1000, // Number of samples per gradient update
        epochs: 200, //larger number of iterations on a non-linear model
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
    let lossTensor =  model.evaluate(testingInputTensor, testingLabelTensor); // we get a tensor
    let loss = await lossTensor[0].dataSync(); // lossTensor is a scalar[], so we take first element

    console.log(`Testing set loss = ${loss}`);
}

async function predict(model, km, age){
    if (isNaN(km) || isNaN(age)){
        console.log("Error: kilometers and age must be valid numbers");
        return null;
    }



    let inputTensor = tf.tensor2d([[km,  age]]);
    //Needs to be normalized because so is model data
    let normalizedInput = normalizeInputs(inputTensor);
    // normalizedInput.print();

    let normalizedOutputTensor = model.predict(normalizedInput);
    let outputTensor = denormalizeOutput(normalizedOutputTensor);
    // output.print();

    let outputValue = outputTensor.dataSync();

    inputTensor.dispose();
    normalizedInput.dispose();
    normalizedOutputTensor.dispose();
    outputTensor.dispose();

    return(outputValue);
}

async function runPredictions(model){
    let features = [
        {km: 9000,   age:1, target:32180},
        {km:100000,  age:5, target:16689},
        {km:50000,   age:1, target: 21838},
        {km:1500,    age:1, target:31352},
        {km:150000,  age:10, target:8919},
        {km:200000,  age:10, target:6236},
        {km:200000,  age:12, target:6327},
        {km:168000,  age:5, target:11654},
    ];

    for (let i=0; i<features.length; i++){
        //predPrice is an object
        let predPrice = await predict (model, features[i].km,   features[i].age);
        let price = predPrice["0"];
        let diffrate = ((price - features[i].target)*100/features[i].target).toFixed(3);
        show(`km=${features[i].km}, age=${features[i].age}, predicted price: ${price.toFixed(0)}, target: ${features[i].target} diff: ${diffrate}%`);
    }
}




//Split data into training set and test set according to threshold value
function splitData(dataArray, threshold){
    let rows = dataArray.length;
    let trainRows = Math.floor(rows * threshold);
    let trainingSet = dataArray.slice(0, trainRows);
    let testingSet = dataArray.slice(trainRows);

    return [trainingSet, testingSet];
}





function normalizeInputs(inputTensor, mode=null){
    // Split into 3 separate tensors - one of each input feature
    let [km, age] = tf.split(inputTensor, 2, 1); // split on axis=1 i.e. vertically
    //Find min and max of 1st and 3rd tensors (before training only)
    if (mode == 'training'){
        metadata.max_km =  km.max();
        metadata.min_km =  km.min();
        metadata.max_age = age.max();
        metadata.min_age = age.min();
    }

    //Normalize 1st and 3rd tensors
// metadata.min_km.print();
    let normalizedKm = km.sub(metadata.min_km).div(metadata.max_km.sub(metadata.min_km)); // (value - min) / (max - min)

    let normalizedAge = age.sub(metadata.min_age).div(metadata.max_age.sub(metadata.min_age)); // (value - min) / (max - min)

    //Concatenate again the three tensors according to dimension 1 ("vertically")
    let normalizedInputTensor = tf.concat([normalizedKm, normalizedAge ], 1);

    // normalizedInputTensor.print();
    return normalizedInputTensor;

}


function denormalizeOutput(outputTensor){
    let normalizedOutputTensor = outputTensor.mul(metadata.max_price.sub(metadata.min_price)).add(metadata.min_price);
    return normalizedOutputTensor;
}


function normalizeLabels(labelTensor){
    metadata.max_price = labelTensor.max();
    metadata.min_price = labelTensor.min();

    let normalizedLabelTensor = labelTensor.sub(metadata.min_price).div(metadata.max_price.sub(metadata.min_price)); // (value - min) / (max - min)

    return normalizedLabelTensor;
}

async function saveModel(model){
    await model.save(modelSaveFileUrl);
    await saveMetadata();
}

async function loadModel(){
    const model = await tf.loadLayersModel(modelLoadFileUrl);
    await loadMetadata();
    return model;
}

async function saveMetadata (){
    //Every element of metadata is a tensor
    let metajson = {
        max_km: {
            data: metadata.max_km.arraySync(),
            shape: metadata.max_km.shape
        },
        min_km: {
            data: metadata.min_km.arraySync(),
            shape: metadata.min_km.shape
        },
        max_age: {
            data: metadata.max_age.arraySync(),
            shape: metadata.max_age.shape
        },
        min_age: {
            data: metadata.min_age.arraySync(),
            shape: metadata.min_age.shape
        },
        max_price: {
            data: metadata.max_price.arraySync(),
            shape: metadata.max_price.shape
        },
        min_price: {
            data: metadata.min_price.arraySync(),
            shape: metadata.min_price.shape
        },
    }

    fs.writeFileSync(metadataFileUrl, JSON.stringify(metajson), 'utf8',
                function(err) {
                    if (err) {
                        console.log('Error saving metadata: ' + err);
                        throw err;
                    }
                }
    );

}

async function loadMetadata(){
    //Every element of metadata is a tensor

    let data = fs.readFileSync(metadataFileUrl, 'utf8',
        function(err) {
            if (err) {
                console.log('Error loading metadata: ' + err);
                throw err;
            }
        }
    );
    let metajson =  JSON.parse(data);

    metadata.max_km = tf.tensor(metajson.max_km.data, metajson.max_km.shape);
    metadata.min_km = tf.tensor(metajson.min_km.data, metajson.min_km.shape);
    metadata.max_age = tf.tensor(metajson.max_age.data, metajson.max_age.shape);
    metadata.min_age = tf.tensor(metajson.min_age.data, metajson.min_age.shape);
    metadata.max_price = tf.tensor(metajson.max_price.data, metajson.max_price.shape);
    metadata.min_price = tf.tensor(metajson.min_price.data, metajson.min_price.shape);

}

function getCurrentTime(){
    let d = new Date();
    let t = d.getTime();
    return (t);
}
