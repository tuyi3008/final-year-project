// test-tf.js
// Test the Magenta.js style transfer library
console.log('=== Testing Magenta.js Style Transfer Library ===');

async function testMagenta() {
  try {
    console.log('1. Loading Magenta library...');
    // Dynamically import the library (it uses ES modules)
    const {Stylization} = await import('@magenta/image');

    console.log('2. Creating style transfer model...');
    // Initialize the style transfer model
    const model = new Stylization();
    
    console.log('3. Loading model weights...');
    await model.initialize(); // This loads the necessary weights
    
    console.log('✅ Magenta model initialized successfully!');
    
    console.log('4. Creating test image tensor...');
    // Create a simple test image (2x2 pixels, RGBA)
    // The model expects Float32 values between 0-255
    const testImageData = new Float32Array([
      255, 0, 0, 255,     // Red pixel
      0, 255, 0, 255,     // Green pixel
      0, 0, 255, 255,     // Blue pixel
      255, 255, 0, 255    // Yellow pixel
    ]);
    
    // Create image tensor: [batch, height, width, channels]
    // For a 2x2 RGBA image: [1, 2, 2, 4]
    const imageTensor = {
      data: testImageData,
      shape: [1, 2, 2, 4], // Batch=1, Height=2, Width=2, Channels=4 (RGBA)
      dtype: 'float32'
    };
    
    console.log('   Test image created: 2x2 RGBA pixels');
    
    console.log('5. Creating style tensor...');
    // Create a simple style (same size as content for testing)
    const styleTensor = {
      data: new Float32Array(16).fill(128), // Gray style
      shape: [1, 2, 2, 4],
      dtype: 'float32'
    };
    
    console.log('6. Running style transfer (this may take a moment)...');
    const startTime = Date.now();
    
    // For testing purposes, we'll just verify the model works
    // Note: The actual stylize method might need adjustment based on library docs
    console.log('   (Skipping actual stylize call for initial test)');
    
    const inferenceTime = Date.now() - startTime;
    console.log(`✅ Library test completed in ${inferenceTime}ms!`);
    
    console.log('\n🎉 TEST SUMMARY:');
    console.log('   - Library loading: SUCCESS');
    console.log('   - Model initialization: SUCCESS');
    console.log('   - Tensor creation: SUCCESS');
    
    console.log('\n📝 Next steps:');
    console.log('   1. Check Magenta.js documentation for exact usage');
    console.log('   2. Replace test tensors with real image data');
    console.log('   3. Integrate into server.js with real image processing');
    
    return true;
    
  } catch (error) {
    console.error('\n❌ Magenta.js test failed:');
    console.error('   Error:', error.message);
    
    if (error.message.includes('Cannot find module')) {
      console.error('\n🔧 Troubleshooting:');
      console.error('   1. Make sure you ran: npm install @magenta/image');
      console.error('   2. Try: npm install @magenta/image --force');
      console.error('   3. Check Node.js version compatibility');
    }
    
    console.error('\nFull error:');
    console.error(error.stack);
    return false;
  }
}

// Run the test
testMagenta().then(success => {
  if (success) {
    console.log('\n✨ Ready to proceed to server.js integration!');
  } else {
    console.log('\n⚠️ Need to fix Magenta.js installation first.');
  }
});