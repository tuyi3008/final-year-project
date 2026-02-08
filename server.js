require('dotenv').config();

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const sharp = require('sharp');

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

        // 1. Preprocess the image with Sharp
        const preprocessedBuffer = await sharp(inputPath)
            .resize(512, 512, { fit: 'inside', withoutEnlargement: true }) // Resize to a common size
            .toFormat('png') // Convert to PNG for consistent format
            .toBuffer();

        // 2. Prepare the API request to Replicate
        const apiToken = process.env.REPLICATE_API_TOKEN;
        if (!apiToken) {
            throw new Error('REPLICATE_API_TOKEN is not set in environment variables');
        }
        const modelVersion = 'stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf'; // A popular img2img model

        const inputData = {
            version: modelVersion,
            input: {
                prompt: `high-quality image in ${style} style`, // Tell the AI the style
                image: preprocessedBuffer.toString('base64'), // Send image as base64
                strength: 0.7, // How much to change the image (0-1)
                guidance_scale: 7.5 // How closely to follow the prompt
            }
        };

        // 3. Make the API call
        const apiResponse = await axios.post(
            'https://api.replicate.com/v1/predictions',
            inputData,
            {
                headers: {
                    'Authorization': `Token ${apiToken}`,
                    'Content-Type': 'application/json',
                    'Prefer': 'wait' // Wait for the result if possible
                }
            }
        );

        // 4. Handle the API response
        //    Replicate's API is asynchronous. We get a status URL.
        const predictionId = apiResponse.data.id;
        const statusUrl = `https://api.replicate.com/v1/predictions/${predictionId}`;

        let resultUrl = null;
        let attempts = 0;
        const maxAttempts = 30; // Try for about 30 seconds

        // Poll the status endpoint until processing is complete
        while (attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
            attempts++;

            const statusCheck = await axios.get(statusUrl, {
                headers: { 'Authorization': `Token ${apiToken}` }
            });

            const status = statusCheck.data.status;

            if (status === 'succeeded') {
                resultUrl = statusCheck.data.output[0]; // The URL of the generated image
                break;
            } else if (status === 'failed' || status === 'canceled') {
                throw new Error(`AI processing failed with status: ${status}`);
            }
            // If still 'processing' or 'starting', loop continues
        }

        if (!resultUrl) {
            throw new Error('AI processing timed out.');
        }

        // 5. Download the generated image
        const imageResponse = await axios.get(resultUrl, { responseType: 'arraybuffer' });
        const generatedImageBuffer = Buffer.from(imageResponse.data, 'binary');

        // 6. Save the final image using Sharp (optional: do final resize/format)
        await sharp(generatedImageBuffer)
            .toFile(outputPath);

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