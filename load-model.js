const tf = require('@tensorflow/tfjs');
// Pure JS environment does not require tfjs-node binary

async function loadModel() {
  try {
    console.log('Loading TensorFlow model...');

    const model = await tf.loadGraphModel(
      'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/tfjs/1',
      { fromTFHub: true }
    );

    console.log('✅ Model loaded!');
    console.log('Inputs:', model.inputs.map(i => i.name));
    console.log('Outputs:', model.outputs.map(o => o.name));

  } catch (err) {
    console.error('❌ Failed to load model');
    console.error(err);
  }
}

loadModel();
