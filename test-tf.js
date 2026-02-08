// test-tf.js
// Test whether the TensorFlow.js Node.js environment is working properly
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

console.log('=== Starting TensorFlow.js environment test ===');

async function test() {
  try {
    // 1. Test TensorFlow.js version and basic functionality
    console.log('1. TensorFlow.js version:', tf.version.tfjs);
    console.log('2. Backend type:', tf.getBackend());

    // 2. Create simple tensors for calculation testing
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6], [7, 8]]);
    const result = a.add(b);
    console.log('3. Tensor calculation test:');
    console.log('   a + b =');
    result.print();

    // 3. Simulate an image tensor (RGB, 512x512)
    console.log('4. Simulating image tensor (1, 512, 512, 3)...');
    const mockImageTensor = tf.randomNormal([1, 512, 512, 3]);
    console.log('   Tensor shape:', mockImageTensor.shape);
    console.log('   Data type:', mockImageTensor.dtype);

    console.log('✅ All basic tests passed! TensorFlow.js environment is working correctly.');
    console.log('Next step: try loading a real style transfer model.');

    // Clean up memory
    a.dispose();
    b.dispose();
    result.dispose();
    mockImageTensor.dispose();

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error);
  }
}

test();
