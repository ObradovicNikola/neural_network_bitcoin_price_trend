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

  // model.add(
  //   tf.layers.lstm({
  //     inputShape: [60, 4],
  //     units: 12,
  //     returnSequences: true,
  //     activation: "tanh",
  //   })
  // );

  // model.add(
  //   tf.layers.lstm({
  //     units: 12,
  //     returnSequences: false,
  //     activation: "tanh",
  //   })
  // );

  // model.add(tf.layers.dense({ units: 32, activation: "sigmoid" }));

  // model.add(tf.layers.dense({ units: 1, activation: "softmax" }));

  // Add a single input layer
  model.add(tf.layers.flatten({ inputShape: [60, 4] }));

  model.add(tf.layers.dense({ units: 40, activation: "sigmoid" }));
  model.add(tf.layers.dense({ units: 40, activation: "sigmoid" }));
  model.add(tf.layers.dense({ units: 20, activation: "sigmoid" }));

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
    for (let i = 0; i < data.length - 60 - 3; i++) {
      let input = {};
      input.xs = data.slice(i, i + 60);
      const lastPrice = data[i + 59].closePrice;
      let higherPriceCount = 0;
      for (let j = i + 60; j <= i + 60 + 3; j++) {
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

    // Step 1. Shuffle the data
    // tf.util.shuffle(rawInputs);
    // console.log("rawInputs shuffled: ", rawInputs);
    // Can't shuffle sequential data!!!
    // test samples would have very similar training samples
    // take a % chunk for testing
    // for time series:
    // take last few % for testing, after that you can shuffle

    // Step 2. Convert data to Tensor and normalize the data to range 0 - 1

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

    const inputs = rawInputs.map((d) => {
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
    const labels = rawInputs.map((d) => d.ys);

    console.log("inputs: ", inputs);
    console.log("labels: ", labels);

    // Balancing
    let num0 = 0;
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] == 0) num0++;
    }
    console.log("Broj nula: ", num0);
    const percentageZeros = ((num0 * 100) / labels.length).toFixed(2);
    console.log(percentageZeros + "% 0s");
    console.log(100 - percentageZeros + "% 1s");
    // it's 55% 1s and 45% 0s
    // data is already balanced

    const trainingInputs = inputs.slice(0, inputs.length * 0.95);
    const testInputs = inputs.slice(inputs.length * 0.95, inputs.length);
    const trainingLabels = labels.slice(0, labels.length * 0.95);
    const testLabels = labels.slice(labels.length * 0.95, labels.length);
    console.log("training inputs length", trainingInputs.length);
    console.log("test inputs length", testInputs.length);
    console.log("training labels length", trainingLabels.length);
    console.log("test labels length", testLabels.length);

    const trainingInputsTensor = tf.tensor3d(trainingInputs, [
      trainingInputs.length,
      60,
      4,
    ]);
    const trainingLabelsTensor = tf.tensor2d(trainingLabels, [
      trainingLabels.length,
      1,
    ]);
    const testInputsTensor = tf.tensor3d(testInputs, [
      testInputs.length,
      60,
      4,
    ]);

    const testLabelsTensor = tf.tensor2d(testLabels, [testLabels.length, 1]);

    console.log("training inputs: ", trainingInputsTensor);
    console.log("test inputs: ", testInputsTensor);

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
  const learningRate = 0.01;
  const learningRate2 = 0.05;
  const decayRate = 0.001;
  model.compile({
    optimizer: tf.train.adam(learningRate),
    // optimizer: tf.train.adam(learningRate2, decayRate),
    loss: tf.losses.meanSquaredError,
    metrics: ["accuracy"],
  });

  //   890 inputs
  //   1, 2, 5, 10, 89, 178, 445, 890
  const batchSize = 10;
  const epochs = 150;

  return await model.fit(trainingInputs, trainingLabels, {
    batchSize,
    epochs,
    validationData: [testInputs, testLabels],
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "acc", "val_loss", "val_acc"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
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
}

document.addEventListener("DOMContentLoaded", run);
