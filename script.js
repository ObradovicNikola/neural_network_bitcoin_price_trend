const INPUT_INTERVAL = 20;
const TRAINING_PERCENTAGE = 0.9;

const LEARNING_RATE = 0.01;
const LEARNING_RATE2 = 0.003;
const DECAY_RATE = 0.0004;
const BATCH_SIZE = 10;
const EPOCHS = 150;

async function getData() {
  let formattedData = [];
  ohlcv.forEach((element) => {
    formattedData.push({
      time: element[0],
      closePrice: element[4],
      volume: element[5],
      quoteVolume: element[6],
    });
  });

  return formattedData;
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  // model.add(
  //   tf.layers.lstm({
  //     inputShape: [INPUT_INTERVAL, 4],
  //     units: 40,
  //     returnSequences: false,
  //     activation: "tanh",
  //   })
  // );
  model.add(tf.layers.flatten({ inputShape: [INPUT_INTERVAL, 4] }));
  
  model.add(tf.layers.dense({ units: 40, activation: "sigmoid" }));
  model.add(tf.layers.dropout(0.2));
  model.add(tf.layers.dense({ units: 40, activation: "sigmoid" }));
  model.add(tf.layers.dropout(0.2));
  model.add(tf.layers.dense({ units: 20, activation: "sigmoid" }));
  model.add(tf.layers.dropout(0.2));

  // Add an output layer
  model.add(tf.layers.dense({ units: 1 }));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Create raw inputs (xs) and labels (ys)
    let rawInputs = [];
    for (let i = 0; i < data.length - INPUT_INTERVAL - 3; i++) {
      let input = {};
      input.xs = data.slice(i, i + INPUT_INTERVAL);
      const lastPrice = data[i + INPUT_INTERVAL - 1].closePrice;
      let higherPriceCount = 0;
      for (let j = i + INPUT_INTERVAL; j <= i + INPUT_INTERVAL + 3; j++) {
        if (data[j].closePrice > lastPrice) {
          higherPriceCount++;
        }
      }
      if (higherPriceCount >= 2) {
        input.ys = 1;
      } else {
        input.ys = 0;
      }
      rawInputs.push(input);
    }

    let rawInputsTraining = rawInputs.slice(
      0,
      rawInputs.length * TRAINING_PERCENTAGE
    );
    let rawInputsTesting = rawInputs.slice(
      rawInputs.length * TRAINING_PERCENTAGE,
      rawInputs.length
    );

    tf.util.shuffle(rawInputsTraining);
    console.log("Raw inputs: ", rawInputsTraining);

    // Balancing
    let num0 = 0;
    let num1 = 0;
    for (let i = 0; i < rawInputsTraining.length; i++) {
      if (rawInputsTraining[i].ys == 0) num0++;
      if (rawInputsTraining[i].ys == 1) num1++;
    }
    console.log("Broj nula: ", num0);
    console.log("Broj jedinica: ", num1);
    let diff = Math.abs(num0 - num1);
    console.log("diff " + diff);

    let k = 0;
    while (diff > 0) {
      if (num0 > num1) {
        if (rawInputsTraining[k].ys == 0) {
          rawInputsTraining.splice(k, 1);
          console.log("brisem nulu");
          diff--;
          k--;
        }
      } else if (num1 > num0) {
        if (rawInputsTraining[k].ys == 1) {
          rawInputsTraining.splice(k, 1);
          console.log("brisem jedinicu");
          diff--;
          k--;
        }
      }
      k++;
    }

    num0 = 0;
    num1 = 0;
    for (let i = 0; i < rawInputsTraining.length; i++) {
      if (rawInputsTraining[i].ys == 0) num0++;
      if (rawInputsTraining[i].ys == 1) num1++;
    }
    console.log("Broj nula: ", num0);
    console.log("Broj jedinica: ", num1);
    diff = Math.abs(num0 - num1);
    console.log("diff " + diff);
    tf.util.shuffle(rawInputsTraining);

    const percentageZeros = ((num0 * 100) / rawInputsTraining.length).toFixed(
      2
    );
    console.log(percentageZeros + "% 0s");
    console.log(100 - percentageZeros + "% 1s");

    // Normalize the data to range 0 - 1
    let inputMin = {
      time: Number.MAX_SAFE_INTEGER,
      closePrice: Number.MAX_SAFE_INTEGER,
      volume: Number.MAX_SAFE_INTEGER,
      quoteVolume: Number.MAX_SAFE_INTEGER,
    };

    let inputMax = {
      time: 0,
      closePrice: 0,
      volume: 0,
      quoteVolume: 0,
    };

    data.forEach((d) => {
      if (d.time < inputMin.time) inputMin.time = d.time;
      if (d.time > inputMax.time) inputMax.time = d.time;

      if (d.closePrice < inputMin.closePrice)
        inputMin.closePrice = d.closePrice;
      if (d.closePrice > inputMax.closePrice)
        inputMax.closePrice = d.closePrice;

      if (d.volume < inputMin.volume) inputMin.volume = d.volume;
      if (d.volume > inputMax.volume) inputMax.volume = d.volume;

      if (d.quoteVolume < inputMin.quoteVolume)
        inputMin.quoteVolume = d.quoteVolume;
      if (d.quoteVolume > inputMax.quoteVolume)
        inputMax.quoteVolume = d.quoteVolume;
    });

    console.log(inputMin);
    console.log(inputMax);

    let inputMinMaxDiff = {
      time: inputMax.time - inputMin.time,
      closePrice: inputMax.closePrice - inputMin.closePrice,
      volume: inputMax.volume - inputMin.volume,
      quoteVolume: inputMax.quoteVolume - inputMin.quoteVolume,
    };

    const inputsTraining = rawInputsTraining.map((d) => {
      return d.xs.map((x) => {
        return Object.values({
          time: (x.time - inputMin.time) / inputMinMaxDiff.time,
          closePrice:
            (x.closePrice - inputMin.closePrice) / inputMinMaxDiff.closePrice,
          volume: (x.volume - inputMin.volume) / inputMinMaxDiff.volume,
          quoteVolume:
            (x.quoteVolume - inputMin.quoteVolume) /
            inputMinMaxDiff.quoteVolume,
        });
      });
    });
    const labelsTraining = rawInputsTraining.map((d) => d.ys);

    const inputsTesting = rawInputsTesting.map((d) => {
      return d.xs.map((x) => {
        return Object.values({
          time: (x.time - inputMin.time) / inputMinMaxDiff.time,
          closePrice:
            (x.closePrice - inputMin.closePrice) / inputMinMaxDiff.closePrice,
          volume: (x.volume - inputMin.volume) / inputMinMaxDiff.volume,
          quoteVolume:
            (x.quoteVolume - inputMin.quoteVolume) /
            inputMinMaxDiff.quoteVolume,
        });
      });
    });
    const labelsTesting = rawInputsTesting.map((d) => d.ys);

    console.log("inputs: ", inputsTraining);
    console.log("labels: ", labelsTraining);

    console.log("training inputs length", inputsTraining.length);
    console.log("test inputs length", inputsTesting.length);
    console.log("training labels length", labelsTraining.length);
    console.log("test labels length", labelsTesting.length);

    const trainingInputsTensor = tf.tensor3d(inputsTraining, [
      inputsTraining.length,
      INPUT_INTERVAL,
      4,
    ]);
    const trainingLabelsTensor = tf.tensor2d(labelsTraining, [
      labelsTraining.length,
      1,
    ]);
    const testInputsTensor = tf.tensor3d(inputsTesting, [
      inputsTesting.length,
      INPUT_INTERVAL,
      4,
    ]);

    const testLabelsTensor = tf.tensor2d(labelsTesting, [
      labelsTesting.length,
      1,
    ]);

    return {
      trainingInputsTensor,
      trainingLabelsTensor,
      testInputsTensor,
      testLabelsTensor,
    };
  });
}

async function trainModel(
  model,
  trainingInputs,
  trainingLabels,
  testInputs,
  testLabels
) {
  // Prepare the model for training.
  model.compile({
    // optimizer: tf.train.adam(LEARNING_RATE),
    optimizer: tf.train.adam(LEARNING_RATE2, DECAY_RATE),
    loss: tf.losses.meanSquaredError,
    metrics: ["accuracy"],
  });

  return await model.fit(trainingInputs, trainingLabels, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    validationData: [testInputs, testLabels],
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "acc", "val_loss", "val_acc"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

function makePrediction(model, x, y) {
  console.log("Prediction:");
  console.log("x: " + x);
  console.log("Correct answer: " + y);
  const xTensor = tf.tensor3d(tf.util.flatten(x), [1, INPUT_INTERVAL, 4]);
  console.log("Model prediction: " + model.predict(xTensor).arraySync()[0]);
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  console.log("Loading the data...");
  const data = await getData();
  console.log(data);
  const values = data.map((d) => ({
    x: d.time,
    y: d.closePrice,
  }));
  console.log("Data loaded.");

  tfvis.render.linechart(
    { name: "BTC/USD" },
    { values },
    {
      xLabel: "Time",
      yLabel: "Price",
      height: 300,
      xAxisDomain: [1624572660, 1624632600],
      yAxisDomain: [32000, 36000],
    }
  );

  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);
  console.log("Model created.");

  // Convert the data to a form we can use for training.
  console.log("Converting data to tensors...");
  console.log(data);
  const tensorData = convertToTensor(data);
  const {
    trainingInputsTensor,
    trainingLabelsTensor,
    testInputsTensor,
    testLabelsTensor,
  } = tensorData;
  console.log("Data ready.");
  // Train the model
  console.log("Training the model...");
  await trainModel(
    model,
    trainingInputsTensor,
    trainingLabelsTensor,
    testInputsTensor,
    testLabelsTensor
  );
  console.log("Done Training.");
  makePrediction(
    model,
    testInputsTensor.arraySync()[0],
    testLabelsTensor.arraySync()[0]
  );
}

document.addEventListener("DOMContentLoaded", run);
