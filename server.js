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

        console.log(`[API Call] Processing: ${req.file.filename}, Style: ${style}`);

        setTimeout(() => {
            res.json({
                success: true,
                originalUrl: `/uploads/${req.file.filename}`,
                processedUrl: `/processed/${filename}`,
                note: 'AI integration in progress. This is a placeholder.'
            });
        }, 500);

    } catch (error) {
        console.error('[Route Error]', error);
        res.status(500).json({ error: 'Server processing failed' });
    }
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});