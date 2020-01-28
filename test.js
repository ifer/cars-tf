const tf = require('@tensorflow/tfjs-node');
const fs = require ('fs');

const tools = require('./tools.js');
const log = tools.log;
const show = tools.show;
const showtable = tools.showtable;

tf.scalar(3.14).print();
