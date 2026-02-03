const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const Jimp = require('jimp'); // used for image processing simulation

const app = express();
const PORT = 3000;

// 1. set up static folders
app.use(express.static('public'));
app.use('/uploads', express.static('uploads'));
app.use('/processed', express.static('processed'));

// 2. ensure upload and processed directories exist
const uploadDir = './uploads';
const processedDir = './processed';
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);
if (!fs.existsSync(processedDir)) fs.mkdirSync(processedDir);

// 3. set up multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});
const upload = multer({ storage: storage });

// 4. core API endpoint for image transformation
app.post('/api/transform', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image uploaded' });
        }

        const style = req.body.style || 'sketch';
        const inputPath = req.file.path;
        const filename = 'processed_' + req.file.filename;
        const outputPath = path.join(processedDir, filename);

        console.log(`Processing image: ${req.file.filename} with style: ${style}`);

        // --- Simulated image processing logic ---
        // in finakl version, replace this with actual style transfer model inference
        const image = await Jimp.read(inputPath);

        if (style === 'sketch') {
            // simulate sketch: desaturate + increase contrast + posterize
            image.greyscale().contrast(0.8).posterize(4);
        } else if (style === 'anime') {
            // simulate anime: increase saturation + slight brightness
            image.color([{ apply: 'saturate', params: [50] }]).brightness(0.1);
        } else if (style === 'ink') {
            // simulate ink: desaturate + blur + decrease brightness
            image.greyscale().blur(2).brightness(-0.1);
        } else {
            // default: invert colors
            image.invert();
        }

        await image.writeAsync(outputPath);
        // -------------------------------------------

        // simulate network delay
        setTimeout(() => {
            res.json({
                success: true,
                originalUrl: `/uploads/${req.file.filename}`,
                processedUrl: `/processed/${filename}`
            });
        }, 2000);

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Processing failed' });
    }
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});