const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const yargs = require('yargs');

const tools = require('./tools.js');
const log = tools.log;
const show = tools.show;
const showtable = tools.showtable;

const dev_train_file = 'data/mnist_train_100.csv';
const dev_test_file = 'data/mnist_test_10.csv';
const prod_train_file = 'data/mnist_train.csv';
const prod_test_file = 'data/mnist_test.csv';
const predictions_file = 'data/mnist_train_100.csv';

const mode = 'PROD';


const modelSaveFileUrl = 'file://saved/mnist_model';
const modelLoadFileUrl = 'file://saved/mnist_model/model.json';
const metadataFileUrl = 'saved/mnist_metadata.json';

const input_nodes = 784;
const hidden_nodes = 200;
const output_nodes = 10;
const learning_rate = 0.1;
const epochs = 50;

let metadata = {};

const arrSum = arr => arr.reduce((a, b) => a + b, 0);

//Syntax: node mnist: creates, trains, tests and saves a model
//        node mnist load: loads the last saved model and runs predictions

const argv = yargs
    .command('save', 'Save model after training and testing')
    .command('load', 'Run a saved model')
    .help()
    .argv;


var save = false;

var command = argv._[0];

if (command === 'load') {
    runSavedModel();
} else {
    if (command === 'save') {
        save = true;
        runSavedModel();
    }
    run();
}

async function runSavedModel() {
    model = await loadModel();
    await runPredictions(model);
}


async function run() {


    let [trainingFeatures, trainingLabels] = await getData('TRAINING');
    // testData.print();


    let model = createModel();
    // model.summary();


    let result = await trainModel(model, trainingFeatures, trainingLabels);
    // console.log(result);

    let [testingFeatures, testingLabels] = await getData('TESTING');
    await testModel(model, testingFeatures, testingLabels);

    if (save) {
        await saveModel(model);
    }

    await runPredictions(model);
}

function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single hidden layer with 10 nodes, expecting 3 inputs and having bias
    model.add(tf.layers.dense({
        inputShape: [input_nodes],
        units: hidden_nodes,
        activation: "sigmoid", // non-linear activation function
        useBias: true
    }));


    // Add an output layer with bias
    model.add(tf.layers.dense({
        units: output_nodes,
        activation: "sigmoid", // softmax activation function: all predictions adapt to a total of 1
        // so that they can be interpreted as percentages (probabilities)
        useBias: true
    }));


    const loss = "categoricalCrossentropy"; // Selected loss function: Categorical Cross Entropy algorithm, suitable for multi-class classification )

    //Optimizer which is going to find minimum loss
    /* const optimizer = tf.train.sgd(0.1); // Selected optimizer: Stochastic Gradient Descent with learning rate 0.1 */
    const optimizer = tf.train.sgd(learning_rate); // Preferred optimizer: the most recently developed and the most efficient

    //// Compile model //// (necessary before training)
    model.compile({
        loss: loss,
        optimizer: optimizer,
        metrics: ['accuracy'],
    });

    return model;
}

async function trainModel(model, trainingInputTensor, trainingLabelTensor) {

    console.log('TRAINING...');

    var t0, t1; // Time holders

    //Training method: fit (returns a promise)
    return model.fit(trainingInputTensor, trainingLabelTensor, {
        batchsize: 1000, // Number of samples per gradient update
        epochs: epochs, //larger number of iterations on a non-linear model
        validationSplit: 0.2, // fraction of the training data to be used as validation data.
        // The model will set apart this fraction of the training data, will not train on it,
        // and will evaluate the loss and any model metrics on this data at the end of each epoch
        callbacks: {
            //
            onTrainBegin: function(logs) {
                t0 = getCurrentTime();
                console.log("TRAINING STARTED");
            },
            onTrainEnd: function(logs) {
                t1 = getCurrentTime();
                let timeElapsed = ((t1 - t0) / 60000).toFixed(2); //in minutes
                console.log(`TRAINING FINISHED in ${timeElapsed} minutes`);
            },

            // On end of each epoch and of batch, print epoch number and loss (visual version)
            // onEpochEnd,
            // On end of each epoch print epoch number and loss (console version)
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss} accuracy = ${log.acc}`),

            onEpochBegin: async function() {}
        }
    });
}

async function testModel(model, testingInputTensor, testingLabelTensor) {

    console.log('TESTING...');
    //Testing the model
    let lossTensor = model.evaluate(testingInputTensor, testingLabelTensor); // we get a tensor
    let loss = await lossTensor[0].dataSync(); // lossTensor is a scalar[], so we take first element

    console.log(`Testing set loss = ${loss}`);
}

async function runPredictions(model) {
    console.log('EVALUATING...');

    let scorecard = [];


    let dataset = fs.readFileSync(predictions_file, 'utf-8')
        .split('\n')
        .filter((value, index) => {
            return value != ''
        });


    for (let i = 0; i < dataset.length; i++) {
        if (dataset[i].length == 0) { //empty line
            continue;
        }

        let a = dataset[i].split(',');
        if (a.length != 785) {
            console.log(`Error: number of columns=${a.length} instead of  784 at line ${i}. Aborting..`);
            break;
        }

        correct_label = a[0];

        a.shift(); //remove first element - we already used it
        let inputs = a.map(scale);
        // console.log(arrSum(inputs));
        // continue;

        let featureTensor = tf.tensor2d(inputs, [1, inputs.length]);

        let results = await model.predict(featureTensor).array();
        featureTensor.dispose();
        results = [].concat(...results); // convert 2d array to 1d
        // console.log(`max=${results.indexOf(Math.max(...results))}`);
        let label = results.indexOf(Math.max(...results)); //predicted number is the index of max value
        // show(`correct_label=${correct_label}, network's answer=${label}`);
        if (correct_label == label) {
            scorecard.push(1);
        } else {
            scorecard.push(0);
        }
    }
    // show(scorecard);
    show(`Performance=${arrSum(scorecard)/scorecard.length}`);
}


async function getData(stage) {
    console.log('LOADING DATA...');

    let filename = getDataFilename(stage);

    let lines = fs.readFileSync(filename, 'utf-8')
        .split('\n')
        .filter((value, index) => {
            return value != ''
        });

    // tf.util.shuffle(lines);

    let inputs = [];
    let targets = [];

    for (let i = 0; i < lines.length; i++) {
        if (lines[i].length == 0) { //empty line
            continue;
        }

        let a = lines[i].split(',');
        if (a.length != 785) {
            console.log(`Error: number of columns=${a.length} instead of  784 at line ${i}. Aborting..`);
            break;
        }

        let target = new Array(10).fill(0.01);
        target[parseInt(a[0])] = 0.99;
        targets.push(target);

        a.shift(); //remove first element - we already used it
        let input = a.map(scale);
        inputs.push(input);


        // console.log(`i=${i+1} j=${j+1} i*j=${(i+1)*(j+1)}`);

    }

    lines = null;

    let featureTensor = tf.tensor2d(inputs);
    let labelTensor = tf.tensor2d(targets, [targets.length, 10]);
    inputs = null;
    targets = null;

    return [featureTensor, labelTensor];
}

async function saveModel(model) {
    await model.save(modelSaveFileUrl);
}

async function loadModel() {
    const model = await tf.loadLayersModel(modelLoadFileUrl);
    return model;
}


function getDataFilename(stage) {
    if (mode === 'PROD') {
        return (stage == 'TRAINING') ? prod_train_file : prod_test_file;
    } else {
        return (stage == 'TESTING') ? dev_train_file : dev_test_file;
    }
}

function scale(x) {
    return (((x / 255.0) * 0.99) + 0.01);
}

function getCurrentTime() {
    let d = new Date();
    let t = d.getTime();
    return (t);
}
