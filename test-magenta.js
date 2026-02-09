// test-magenta.js
const tf = require('@tensorflow/tfjs-node');

async function run() {
  try {
    console.log('Loading Magenta TFJS model...');
    const modelUrl = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/tfjs/1';
    const model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });
    console.log('✅ Model loaded!');

    console.log('Creating dummy input tensors...');
    // The model expects two inputs: 'input_1' (content image) and 'input_2' (style image)
    const content = tf.randomNormal([1, 256, 256, 3]); // 256x256 RGB
    const style = tf.randomNormal([1, 256, 256, 3]);

    console.log('Running style transfer...');
    const output = model.predict([content, style]);
    
    console.log('Output tensor shape:', output.shape);
    
    // clean up
    tf.dispose([content, style, output]);
    console.log('✅ Test finished successfully!');

  } catch (err) {
    console.error('❌ Error:', err.message);
  }
}

run();
