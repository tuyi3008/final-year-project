// server.js
// Node.js + Express + tfjs-node backend for multi-style image style transfer

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const PORT = 3000;

// -------------------- Upload Config --------------------
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname))
});

const upload = multer({ storage });

// -------------------- Load Models --------------------
// We keep a map of style name -> tfjs-node model
const styleModels = {};
const styles = ['sketch', 'anime', 'ink']; // Must match your folder names

async function loadModels() {
  for (const style of styles) {
    const modelPath = `file://${path.join(__dirname, 'tfjs_model', style, 'model.json')}`;
    try {
      console.log(`Loading model for style: ${style} ...`);
      const model = await tf.loadGraphModel(modelPath);
      styleModels[style] = model;
      console.log(`✅ Model loaded for style: ${style}`);
    } catch (err) {
      console.error(`❌ Failed to load model for style ${style}:`, err);
    }
  }
}

// Immediately load models
loadModels();

// -------------------- Tensor Conversion Utilities --------------------
async function imageToTensor(filePath) {
  // Read image from disk, convert to [1, height, width, 3] tensor
  const buffer = await fs.promises.readFile(filePath);
  const tensor = tf.node.decodeImage(buffer, 3)
    .expandDims(0)
    .toFloat()
    .div(tf.scalar(255)); // normalize to [0,1]
  return tensor;
}

async function tensorToBuffer(tensor) {
  // Convert output tensor back to JPEG buffer
  const clipped = tensor.clipByValue(0, 1).mul(255).cast('int32');
  const buffer = await tf.node.encodeJpeg(clipped.squeeze());
  return buffer;
}

// -------------------- Express Middlewares --------------------
app.use('/uploads', express.static(uploadDir));
app.use(express.json());

// -------------------- POST /api/transform --------------------
app.post('/api/transform', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.json({ success: false, error: 'No file uploaded' });

    const selectedStyle = req.body.style;
    if (!selectedStyle || !styleModels[selectedStyle]) {
      return res.json({ success: false, error: 'Invalid style selected' });
    }

    const contentTensor = await imageToTensor(req.file.path);
    const model = styleModels[selectedStyle];

    // -------------------- Model Inference --------------------
    // The input/output names depend on your TFJS model structure
    // Usually 'input' -> content image tensor, output is stylized image
    const outputTensor = model.execute({ 'input': contentTensor });

    // Convert tensor to JPEG buffer
    const processedBuffer = await tensorToBuffer(outputTensor);

    const outPath = path.join(uploadDir, `stylized_${selectedStyle}_${Date.now()}.jpg`);
    await fs.promises.writeFile(outPath, processedBuffer);

    // Respond with URL
    res.json({
      success: true,
      processedUrl: `/uploads/${path.basename(outPath)}`
    });

    // Dispose tensors
    contentTensor.dispose();
    outputTensor.dispose();

  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// -------------------- Start Server --------------------
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
